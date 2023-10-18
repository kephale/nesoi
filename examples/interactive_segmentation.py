from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QWidget,
)

from functools import partial
from skimage.feature import multiscale_basic_features

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import zarr

from skimage import future

from psygnal import debounced

from superqt import ensure_main_thread

import toolz as tz

import threading

import napari

import logging
import sys

LOGGER = logging.getLogger("napari.experimental._progressive_loading")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)



class NapariMLWidget(QWidget):
    def __init__(self, parent=None):
        super(NapariMLWidget, self).__init__(parent)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Dropdown for selecting the model
        model_label = QLabel("Select Model")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Random Forest", "XGBoost"])
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        layout.addLayout(model_layout)

        # Select the range of sigma sizes
        self.sigma_start_spinbox = QDoubleSpinBox()
        self.sigma_start_spinbox.setRange(0, 256)
        self.sigma_start_spinbox.setValue(1)

        self.sigma_end_spinbox = QDoubleSpinBox()
        self.sigma_end_spinbox.setRange(0, 256)
        self.sigma_end_spinbox.setValue(5)

        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma Range: From"))
        sigma_layout.addWidget(self.sigma_start_spinbox)
        sigma_layout.addWidget(QLabel("To"))
        sigma_layout.addWidget(self.sigma_end_spinbox)
        layout.addLayout(sigma_layout)

        # Boolean options for features
        self.intensity_checkbox = QCheckBox("Intensity")
        self.intensity_checkbox.setChecked(True)
        self.edges_checkbox = QCheckBox("Edges")
        self.texture_checkbox = QCheckBox("Texture")
        self.texture_checkbox.setChecked(True)

        features_group = QGroupBox("Features")
        features_layout = QVBoxLayout()
        features_layout.addWidget(self.intensity_checkbox)
        features_layout.addWidget(self.edges_checkbox)
        features_layout.addWidget(self.texture_checkbox)
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        # Dropdown for data selection
        data_label = QLabel("Select Data for Model Fitting")
        self.data_dropdown = QComboBox()
        self.data_dropdown.addItems(
            ["Current XY Slice", "Current Displayed Region", "Whole Image"]
        )
        self.data_dropdown.setCurrentText("Current Displayed Region")
        data_layout = QHBoxLayout()
        data_layout.addWidget(data_label)
        data_layout.addWidget(self.data_dropdown)
        layout.addLayout(data_layout)

        # Checkbox for live model fitting
        self.live_fit_checkbox = QCheckBox("Live Model Fitting")
        self.live_fit_checkbox.setChecked(True)
        layout.addWidget(self.live_fit_checkbox)

        # Checkbox for live prediction
        self.live_pred_checkbox = QCheckBox("Live Prediction")
        self.live_pred_checkbox.setChecked(True)
        layout.addWidget(self.live_pred_checkbox)

        self.setLayout(layout)


def extract_features(image, feature_params):
    features_func = partial(
        multiscale_basic_features,
        intensity=feature_params["intensity"],
        edges=feature_params["edges"],
        texture=feature_params["texture"],
        sigma_min=feature_params["sigma_min"],
        sigma_max=feature_params["sigma_max"],
        channel_axis=None,
    )
    print(f"image shape {image.shape} feature params {feature_params}")
    features = features_func(np.squeeze(image))
    return features


# Model training function that respects widget's model choice
def update_model(labels, features, model_type):
    features = features[labels > 0, :]
    # We shift labels - 1 because background is 0 and has special meaning, but models need to start at 0
    labels = labels[labels > 0] - 1
    
    if model_type == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05
        )
    elif model_type == "XGBoost":
        # Adjust for 0 labels
        clf = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=len(np.unique(labels)),
        )

    print(
        f"updating model with label shape  {labels.shape} feature shape {features.shape} unique labels {np.unique(labels)}"
    )
    
    clf.fit(features, labels)

    return clf


def predict(model, features, model_type):
    # We shift labels + 1 because background is 0 and has special meaning
    prediction = future.predict_segmenter(features.reshape(-1, features.shape[-1]), model).reshape(features.shape[:-1]) + 1

    return np.transpose(prediction)


def segment(zarr_path, viewer=None):
    """
    Read a Zarr array.

    Args:
        zarr_path (str): Path to the Zarr array where images will be stored.
    """
    print(f"Reading {zarr_path}")

    # Load Zarr data
    zarr_array = zarr.open(zarr_path, mode='r')

    # TODO is there a NGFF place to find the image name?
    image_layer = viewer.add_image(zarr_array, name=zarr_path)

    # Create a prediction layer
    prediction_data = zarr.open(
        f"{zarr_path}/prediction",
        mode='a',
        shape=zarr_array.shape,
        dtype='i4',
        dimension_separator="/",
        
    )
    prediction_layer = viewer.add_labels(prediction_data, name="Prediction")

    # Use a disk-backed zarr for painting
    # Create a painting layer
    # painting_data = np.zeros_like(image)
    # painting_data = zarr.zeros(zarr_array.shape[:-1], dtype=zarr_array.dtype)
    painting_data = zarr.open(
        f"{zarr_path}/painting",
        mode='a',
        shape=zarr_array.shape,
        dtype='i4',
        dimension_separator="/",
    )
    painting_layer = viewer.add_labels(painting_data, name="Painting")

    model = None

    def threaded_on_data_change(
        event,
        corner_pixels,
        dims,
        model_type,
        feature_params,
        live_fit,
        live_prediction,
        data_choice,
    ):
        global model
        LOGGER.info(f"Labels data has changed! {event}")

        current_step = dims.current_step

        LOGGER.info(f"corner pixels {corner_pixels}")
        
        mask_idx = (slice(viewer.dims.current_step[0], viewer.dims.current_step[0]+1), slice(corner_pixels[0, 1], corner_pixels[1, 1]), slice(corner_pixels[0, 2], corner_pixels[1, 2]))
        if data_choice == "Whole Image":
            mask_idx = tuple([slice(0, sz) for sz in zarr_array.shape])

        LOGGER.info(f"mask idx {mask_idx}, image {image_layer.data.shape}")
        active_image = image_layer.data[mask_idx]
        LOGGER.info(
            f"image shape {active_image.shape} data choice {data_choice} painting_data {painting_data.shape} mask_idx {mask_idx}"
        )

        active_labels = painting_data[mask_idx]

        def compute_features(image, feature_params):
            """Compute features for each channel and concatenate them."""
            list_of_features = []

            for channel in range(image.shape[0]):
                channel_image = np.squeeze(image[channel, ...])
                channel_features = extract_features(
                    channel_image, feature_params
                )
                list_of_features.append(channel_features)

            return np.squeeze(np.stack(list_of_features))

        training_labels = None
        
        if data_choice == "Whole Image":
            LOGGER.info("Processing whole image")
            if not live_fit and live_prediction:
                # Process slice by slice
                LOGGER.info("Predicting for whole image")
                for z in range(image_layer.data.shape[0]):
                    LOGGER.info(f"Predicing whole image for slice {z}")
                    slice_idx = (slice(dims.current_step[0], dims.current_step[0]+1), slice(0, image_layer.data.shape[1]), slice(0, image_layer.data.shape[2]))
                    slice_image = image_layer.data[slice_idx]
                    prediction_features = compute_features(
                        slice_image, feature_params
                    )
                    prediction = predict(model, prediction_features, model_type)
                    LOGGER.info(
                        f"prediction {prediction.shape} prediction layer {prediction_layer.data.shape} prediction {np.transpose(prediction).shape} features {prediction_features.shape}"
                    )
            
                    prediction_data[slice_idx] = np.transpose(prediction)[
                        np.newaxis, :
                    ]
            else:
                # Use the entire image (or all painted labels) for model fitting.
                LOGGER.info("not processing whole image too expensive")
                # training_features = compute_features(
                #     image_layer.data, feature_params
                # )
                # import pdb; pdb.set_trace()
                # LOGGER.info("done computing features")
                # training_labels = np.squeeze(painting_data)

        elif data_choice == "Current Displayed Region":
            # Use only the currently displayed region.
            training_features = compute_features(
                active_image, feature_params
            )
            training_labels = np.squeeze(active_labels)

        elif data_choice == "Current XY Slice":
            # Use only the current XY slice.
            current_slice_idx = tuple(
                dims.current_step[i]
                if i < len(dims.current_step)
                else slice(None)
                for i in range(len(image_layer.data.shape))
            )
            current_slice_image = image_layer.data[current_slice_idx]
            training_features = compute_features(
                current_slice_image, feature_params
            )
            training_labels = np.squeeze(painting_data[current_slice_idx[:-1]])

        else:
            raise ValueError(f"Invalid data choice: {data_choice}")
        
        if (training_labels is None) or (training_labels.shape[0] == 0):
            LOGGER.info("No training data yet. Skipping model update")
        elif live_fit:
            # Retrain model
            LOGGER.info(
                f"training model with labels {training_labels.shape} features {training_features.shape} unique labels {np.unique(training_labels[:])}"
            )
            model = update_model(training_labels, training_features, model_type)

        # Don't do live prediction on whole image, that happens earlier slicewise
        if live_prediction and data_choice != "Whole Image":
            # Update prediction_data
            prediction_features = compute_features(
                active_image, feature_params
            )
            # Add 1 becasue of the background label adjustment for the model
            prediction = predict(model, prediction_features, model_type)
            LOGGER.info(
                f"prediction {prediction.shape} prediction layer {prediction_layer.data.shape} prediction {np.transpose(prediction).shape} features {prediction_features.shape}"
            )
            
            if data_choice == "Whole Image":
                prediction_layer.data[mask_idx] = np.transpose(prediction)
            else:
                prediction_layer.data[mask_idx] = np.transpose(prediction)[
                    np.newaxis, :
                ]

    @tz.curry
    def on_data_change(event, viewer=None, widget=None):
        corner_pixels = image_layer.corner_pixels

        painting_layer.refresh()

        thread = threading.Thread(
            target=threaded_on_data_change,
            args=(
                event,
                corner_pixels,
                viewer.dims,
                widget.model_dropdown.currentText(),
                {
                    "sigma_min": widget.sigma_start_spinbox.value(),
                    "sigma_max": widget.sigma_end_spinbox.value(),
                    "intensity": widget.intensity_checkbox.isChecked(),
                    "edges": widget.edges_checkbox.isChecked(),
                    "texture": widget.texture_checkbox.isChecked(),
                },
                widget.live_fit_checkbox.isChecked(),
                widget.live_pred_checkbox.isChecked(),
                widget.data_dropdown.currentText(),
            ),
        )
        thread.start()
        thread.join()

        prediction_layer.refresh()

    widget = NapariMLWidget()
    viewer.window.add_dock_widget(widget)

    # Adjusted connection to account for the widget
    for listener in [
        viewer.camera.events,
        viewer.dims.events,
        painting_layer.events.paint,
    ]:
        listener.connect(
            debounced(
                ensure_main_thread(
                    on_data_change(
                        viewer=viewer,
                        widget=widget,  # pass the widget instance for easy access to settings
                    )
                ),
                timeout=1000,
            )
        )


if __name__ == "__main__":
    import mrcfile

    zarr_path = "/Users/kharrington/Data/EMPIAR/10548.zarr"

    viewer = napari.Viewer()

    tomo_name = "tomo04"
    tomo_path = f"/Users/kharrington/Data/EMPIAR/10548/archive/10548/data/{tomo_name}.mrc"

    mrc = mrcfile.open(tomo_path, permissive=True)

    # img_layer = viewer.add_image(mrc.data)

    data = mrc.data
    # data = data[250:260, :, :]
    chunk_shape = (128, 128, 128)
    # chunk_shape = (10, 128, 128)

    output = zarr.open(
        zarr_path,
        mode='a',
        shape=data.shape,
        chunks=chunk_shape,
        dtype=data.dtype,
        dimension_separator="/",
    )

    output[:] = data

    segment(zarr_path, viewer=viewer)

"""

Current TODOs:
- implement predictor as zarrstore
- split filaments https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20h_segmentation_post_processing/splitting_touching_objects.html

"""


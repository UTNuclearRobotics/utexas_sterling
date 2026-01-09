import sys
import os
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, GdkPixbuf
from cluster import Cluster, PatchRenderer
from multiprocessing import Pool
import h5py
import gc
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))

def save_image(args):
    """Helper function for parallel image saving."""
    i, image, label, save_path = args
    image_path = os.path.join(save_path, f"{label}.jpg")
    pil_image = Image.fromarray(np.uint8(image))
    pil_image.save(image_path, "JPEG")

class ClusterUI(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.ClusterUI")
        GLib.set_application_name("Terrain Cluster UI")

    def do_activate(self):
        # Create a window
        window = Gtk.ApplicationWindow(application=self, title="Terrain Cluster")
        window.set_resizable(True)
        window.set_default_size(800, 600)

        # Create a scrolled window
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        window.add(scrolled_window)

        # Create a vertical box layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        scrolled_window.add(vbox)

        spf = SelectVicregFile(window)
        vbox.pack_start(spf.get_component(), expand=True, fill=True, padding=0)

        vbox.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), expand=False, fill=True, padding=0)

        ssf = SelectSyncedFile(window)
        vbox.pack_start(ssf.get_component(), expand=True, fill=True, padding=0)

        vbox.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), expand=False, fill=True, padding=0)

        smf = SelectModelFile(window)
        vbox.pack_start(smf.get_component(), expand=True, fill=True, padding=0)

        vbox.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), expand=False, fill=True, padding=0)

        gc = GenerateClusters(window, spf, ssf, smf)
        vbox.pack_start(gc.get_component(), expand=True, fill=True, padding=0)

        # Show the window
        window.show_all()


class SelectVicregFile:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.data_h5_path = None

    def get_component(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)

        vbox.pack_start(Gtk.Label(label="Vicreg .h5 file selected:"), expand=False, fill=True, padding=0)

        # Pickle file status
        self.label = Gtk.Label(label="None")
        vbox.pack_start(self.label, expand=False, fill=True, padding=0)

        # Pickle file chooser button
        file_chooser_button = Gtk.Button(label="Open")
        file_chooser_button.connect("clicked", self.on_file_chooser_button_clicked)
        vbox.pack_start(file_chooser_button, expand=False, fill=True, padding=0)

        return vbox

    def on_file_chooser_button_clicked(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select Data .h5 File",
            transient_for=self.parent_window,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL, "_Open", Gtk.ResponseType.ACCEPT)
        dialog.connect("response", self.on_file_chooser_response)
        dialog.show()

    def on_file_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            file_path = dialog.get_file().get_path()
            if not file_path.endswith(".h5"):
                self.label.set_markup("<span foreground='red'>Error: Selected file is not a .h5 file</span>")
                dialog.destroy()
                return

            self.data_h5_path = file_path
            self.label.set_markup(f"<span foreground='green'>{self.data_h5_path}</span>")
        dialog.destroy()

class SelectSyncedFile:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.synced_h5_path = None

    def get_component(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)

        vbox.pack_start(Gtk.Label(label="Synced .h5 file selected:"), expand=False, fill=True, padding=0)

        # Pickle file status
        self.label = Gtk.Label(label="None")
        vbox.pack_start(self.label, expand=False, fill=True, padding=0)

        # Pickle file chooser button
        file_chooser_button = Gtk.Button(label="Open")
        file_chooser_button.connect("clicked", self.on_file_chooser_button_clicked)
        vbox.pack_start(file_chooser_button, expand=False, fill=True, padding=0)

        return vbox

    def on_file_chooser_button_clicked(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select Data .h5 File",
            transient_for=self.parent_window,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL, "_Open", Gtk.ResponseType.ACCEPT)
        dialog.connect("response", self.on_file_chooser_response)
        dialog.show()

    def on_file_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            file_path = dialog.get_file().get_path()
            if not file_path.endswith(".h5"):
                self.label.set_markup("<span foreground='red'>Error: Selected file is not a .h5 file</span>")
                dialog.destroy()
                return

            self.synced_h5_path= file_path
            self.label.set_markup(f"<span foreground='green'>{self.synced_h5_path}</span>")
        dialog.destroy()


class SelectModelFile:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.model_path = None

    def get_component(self):
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)

        vbox.pack_start(Gtk.Label(label="Model file selected:"), expand=False, fill=True, padding=0)

        # Pickle file status
        self.label = Gtk.Label(label="None")
        vbox.pack_start(self.label, expand=False, fill=True, padding=0)

        # Pickle file chooser button
        file_chooser_button = Gtk.Button(label="Open")
        file_chooser_button.connect("clicked", self.on_file_chooser_button_clicked)
        vbox.pack_start(file_chooser_button, expand=False, fill=True, padding=0)

        return vbox

    def on_file_chooser_button_clicked(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Select PyTorch Model File",
            transient_for=self.parent_window,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL, "_Open", Gtk.ResponseType.ACCEPT)
        dialog.connect("response", self.on_file_chooser_response)
        dialog.show()

    def on_file_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            file_path = dialog.get_file().get_path()
            if not file_path.endswith(".pt"):
                self.label.set_markup("<span foreground='red'>Error: Selected file is not a .pt file</span>")
                dialog.destroy()
                return

            self.model_path = file_path
            self.label.set_markup(f"<span foreground='green'>{self.model_path}</span>")
        dialog.destroy()

class GenerateClusters:
    def __init__(self, parent_window, spf, ssf, smf):
        self.parent_window = parent_window
        self.spf = spf  # Object with data_pkl_path (now vicreg_h5_path)
        self.ssf = ssf  # Object with synced_pkl_path (now synced_h5_path)
        self.smf = smf  # Object with model_path

        self.generated_flag = False
        self.cluster = None
        self.all_cluster_image_indices = None
        self.cluster_labels = None
        self.dataset = None
        self.representation_vectors_np = None

    def get_component(self):
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.vbox.set_margin_top(10)
        self.vbox.set_margin_bottom(10)
        self.vbox.set_margin_start(10)
        self.vbox.set_margin_end(10)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        label = Gtk.Label(label="Number of Clusters:")
        hbox.pack_start(label, expand=False, fill=True, padding=0)

        self.entry_clusters = Gtk.Entry()
        self.entry_clusters.set_placeholder_text("Enter number of clusters...")
        self.entry_clusters.set_text("5")
        hbox.pack_start(self.entry_clusters, expand=False, fill=True, padding=0)
        self.vbox.pack_start(hbox, expand=True, fill=True, padding=0)

        hbox_iterations = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        label_iterations = Gtk.Label(label="Number of Iterations:")
        hbox_iterations.pack_start(label_iterations, expand=False, fill=True, padding=0)

        self.entry_iterations = Gtk.Entry()
        self.entry_iterations.set_placeholder_text("Enter number of iterations...")
        self.entry_iterations.set_text("100")
        hbox_iterations.pack_start(self.entry_iterations, expand=False, fill=True, padding=0)
        self.vbox.pack_start(hbox_iterations, expand=True, fill=True, padding=0)

        button = Gtk.Button(label="Generate Clusters")
        button.connect("clicked", self.on_button_clicked)
        self.vbox.pack_start(button, expand=False, fill=True, padding=0)

        return self.vbox

    def on_button_clicked(self, button):
        vicreg_h5_path = self.spf.data_h5_path  # Now an .h5 path
        synced_h5_path = self.ssf.synced_h5_path  # Now an .h5 path
        model_path = self.smf.model_path

        if not all([vicreg_h5_path, synced_h5_path, model_path]):
            error_dialog = Gtk.MessageDialog(
                transient_for=self.parent_window,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="Please select both the VICReg .h5 file, synced .h5 file, and a model file before generating clusters.",
            )
            error_dialog.show()
            error_dialog.connect("response", lambda dialog, response: dialog.destroy())
            return

        num_clusters = self.entry_clusters.get_text()
        num_iterations = self.entry_iterations.get_text()

        try:
            num_clusters = int(num_clusters)
            num_iterations = int(num_iterations)
            if num_clusters <= 0 or num_iterations <= 0:
                raise ValueError
        except ValueError:
            error_dialog = Gtk.MessageDialog(
                transient_for=self.parent_window,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="Please enter valid positive numbers for clusters and iterations.",
            )
            error_dialog.show()
            error_dialog.connect("response", lambda dialog, response: dialog.destroy())
            return

        save_path = os.path.join(os.path.dirname(vicreg_h5_path), "clusters")
        os.makedirs(save_path, exist_ok=True)

        # Generate clusters with lazy loading
        self.cluster = Cluster(vicreg_h5_path, synced_h5_path, model_path, batch_size=10000)
        self.all_cluster_image_indices = self.cluster.generate_clusters(
            num_clusters,
            num_iterations,
            save_model_path=os.path.join(save_path, "kmeans_model.pkl"),
        )

        sizes = [len(indices) for indices in self.all_cluster_image_indices]
        order = np.argsort(sizes)[::-1]   # descending order

        # Reorder the list of cluster indices
        self.all_cluster_image_indices = [self.all_cluster_image_indices[i] for i in order]

        # Store the dataset (still lazy-loading)
        self.dataset = self.cluster.dataset
        self.cluster_labels = np.zeros(len(self.dataset), dtype=int)
        for cluster_idx, indices in enumerate(self.all_cluster_image_indices):
            for idx in indices:
                self.cluster_labels[idx] = cluster_idx

        self.representation_vectors_np = self.cluster.get_embeddings()
        
        # Target at least 1500 samples per cluster
        min_samples = 1500  # Fixed minimum number of samples to show

        # Render clusters incrementally without storing all patches in memory
        self.images = []
        renderer = PatchRenderer()

        for cluster_indices in self.all_cluster_image_indices:
            # Determine how many samples to take: 1500 or all available if less than 1500
            num_samples = min(min_samples, len(cluster_indices))
            
            # Randomly select indices (without replacement)
            if num_samples < len(cluster_indices):
                selected_indices = np.random.choice(
                    cluster_indices,  # Array/list of indices to sample from
                    size=num_samples,  # Number of indices to select
                    replace=False  # No duplicates
                )
            else:
                selected_indices = cluster_indices  # Use all if fewer than 1500
            
            cluster_patches = []
            # Process randomly selected samples incrementally
            for idx in selected_indices:
                patch1, _, _ = self.dataset[idx]
                # Render patch with RGB input (from VICReg .h5) and RGB output for display
                patch_np = renderer.render_patch(patch1, input_format="BGR", output_format="RGB")
                cluster_patches.append(patch_np)

            # Create a grid for this cluster
            if num_samples > 0:
                num_rows = int(np.ceil(np.sqrt(num_samples)))  # Rough square root for balanced grid
                num_cols = int(np.ceil(num_samples / num_rows))  # Adjust columns to fit all samples
                rendered_cluster = renderer.image_grid(
                    cluster_patches,
                    image_size=(64, 64),
                    output_format="RGB"
                )
                self.images.append(rendered_cluster)
            
            # Clean up to free memory
            del cluster_patches
            gc.collect()

        def numpy_to_pixbuf(array):
            height, width, channels = array.shape
            if channels not in (3, 4):
                raise ValueError("Array must have 3 (RGB) or 4 (RGBA) channels")
            data = array.tobytes()
            rowstride = width * channels
            return GdkPixbuf.Pixbuf.new_from_data(
                data, GdkPixbuf.Colorspace.RGB, channels == 4, 8, width, height, rowstride
            )
        
        hbox_images = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.labels_and_rankings = []
        for i, image in enumerate(self.images):
            pixbuf = numpy_to_pixbuf(image)
            image_widget = Gtk.Image.new_from_pixbuf(pixbuf)
            image_widget.set_size_request(400, 400)

            text_field = Gtk.Entry()
            text_field.set_placeholder_text(f"Label cluster{i + 1}...")
            text_field.set_text(f"cluster{i + 1}")

            ranking_field = Gtk.Entry()
            ranking_field.set_placeholder_text(f"Rank cluster{i + 1}...")
            ranking_field.set_text("0")

            vbox_image = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
            vbox_image.pack_start(
                Gtk.Label(label=f"Cluster {i+1}  ({len(self.all_cluster_image_indices[i])} samples)"),
                expand=False, fill=True, padding=5
            )
            vbox_image.pack_start(image_widget, expand=False, fill=True, padding=0)
            vbox_image.pack_start(Gtk.Label(label=f"Label {i + 1}"), expand=False, fill=True, padding=0)
            vbox_image.pack_start(text_field, expand=False, fill=True, padding=0)
            vbox_image.pack_start(Gtk.Label(label="Preference"), expand=False, fill=True, padding=0)
            vbox_image.pack_start(ranking_field, expand=False, fill=True, padding=0)

            hbox_images.pack_start(vbox_image, expand=True, fill=True, padding=0)
            self.labels_and_rankings.append((text_field, ranking_field))

        # Remove previous separator, hbox_images, and save_button if generated_flag is True
        if self.generated_flag:
            children = self.vbox.get_children()
            if len(children) >= 3:
                for _ in range(3):
                    self.vbox.remove(children[-1])  # Remove last child
                    children = self.vbox.get_children()  # Update children list
        self.generated_flag = True

        self.vbox.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), expand=False, fill=True, padding=0)
        self.vbox.pack_start(hbox_images, expand=True, fill=True, padding=0)

        save_button = Gtk.Button(label="Save Labels and Preferences")
        save_button.connect("clicked", self.on_save_button_clicked)
        self.vbox.pack_start(save_button, expand=False, fill=True, padding=0)

        self.vbox.show_all()

    def on_save_button_clicked(self, button):
        vicreg_h5_path = self.spf.data_h5_path
        save_path = os.path.join(os.path.dirname(vicreg_h5_path), "clusters")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "homography", "config.yaml")
        dataset_save_path = os.path.join(save_path, "labeled_data.h5")

        os.makedirs(save_path, exist_ok=True)
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            if os.path.isfile(file_path) and file_path.endswith(".jpg"):
                os.unlink(file_path)

        user_labels_and_rankings = []
        for text_field, ranking_field in self.labels_and_rankings:
            label = text_field.get_text()
            ranking = ranking_field.get_text()
            try:
                ranking = float(ranking)
            except ValueError:
                error_dialog = Gtk.MessageDialog(
                    transient_for=self.parent_window,
                    modal=True,
                    message_type=Gtk.MessageType.ERROR,
                    buttons=Gtk.ButtonsType.OK,
                    text=f"Invalid preference value: {ranking}. Please enter a valid number.",
                )
                error_dialog.show()
                error_dialog.connect("response", lambda dialog, response: dialog.destroy())
                return
            user_labels_and_rankings.append((label, ranking))

        label_to_new_id = {}
        current_id = 0
        for label, _ in user_labels_and_rankings:
            if label not in label_to_new_id:
                label_to_new_id[label] = current_id
                current_id += 1

        new_cluster_labels = np.zeros_like(self.cluster_labels, dtype=int)
        for cluster_idx, indices in enumerate(self.all_cluster_image_indices):
            user_label = user_labels_and_rankings[cluster_idx][0]
            new_id = label_to_new_id[user_label]
            for idx in indices:
                new_cluster_labels[idx] = new_id

        unique_labels = sorted(label_to_new_id.keys(), key=lambda x: label_to_new_id[x])
        new_terrains = []
        for label in unique_labels:
            for orig_label, preference in user_labels_and_rankings:
                if orig_label == label:
                    new_terrains.append({
                        'name': label,
                        'label': label_to_new_id[label],
                        'preference': preference
                    })
                    break

        cluster_to_terrain = {
            label_to_new_id[label]: {'terrain_label': label, 'preference': preference}
            for label, preference in user_labels_and_rankings
        }

        # Plot clusters with new labels
        self.cluster.plot_clusters_with_labels(
            self.representation_vectors_np,
            new_cluster_labels,
            label_to_new_id,
            len(unique_labels),
            save_plot_path=save_path
        )

        # Parallelize image saving with unique labels
        unique_images = []
        unique_label_prefs = []
        seen_labels = set()
        for i, (label, pref) in enumerate(user_labels_and_rankings):
            if label not in seen_labels:
                unique_images.append(self.images[i])
                unique_label_prefs.append((label, pref))
                seen_labels.add(label)

        num_images = len(unique_images)
        if num_images > 1:  # Only parallelize if worth it
            with Pool(processes=min(os.cpu_count(), num_images)) as pool:
                pool.map(save_image, [(i, unique_images[i], unique_label_prefs[i][0], save_path) 
                                    for i in range(num_images)])
        else:
            for i, image in enumerate(unique_images):
                save_image((i, image, unique_label_prefs[i][0], save_path))

        # Preallocate and batch-write HDF5 data
        num_samples = len(self.dataset)
        with h5py.File(dataset_save_path, 'w') as h5f:
            # Preallocate datasets with known sizes
            patch_shape = (num_samples, 3, 128, 128)  # Adjust if shape varies
            inertial_exists = self.dataset[0][2] is not None
            inertial_shape = self.dataset[0][2].shape if inertial_exists else None
            
            patches_dset = h5f.create_dataset('patches', shape=patch_shape, dtype=np.float32, 
                                            compression='lzf')  # Faster compression
            if inertial_exists:
                inertial_dset = h5f.create_dataset('inertial', shape=(num_samples, *inertial_shape), 
                                                dtype=np.float32, compression='lzf')
            terrain_labels_dset = h5f.create_dataset('terrain_labels', shape=(num_samples,), 
                                                    dtype=h5py.string_dtype(encoding='utf-8'))
            preferences_dset = h5f.create_dataset('preferences', shape=(num_samples,), dtype=np.float32)

            # Batch process and write
            batch_size = 1000  # Adjust based on memory vs. speed trade-off
            for start_idx in tqdm(range(0, num_samples, batch_size), desc="Writing labeled data batches"):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_patches = []
                batch_inertial = [] if inertial_exists else None
                batch_labels = []
                batch_prefs = []

                for idx in range(start_idx, end_idx):
                    patch = self.dataset[idx][0]
                    if isinstance(patch, torch.Tensor):
                        patch = patch.numpy()
                    elif not isinstance(patch, np.ndarray):
                        patch = np.array(patch)
                    if patch.shape[-3:] != (3, 128, 128):
                        patch = patch.transpose(2, 0, 1)

                    inertial = self.dataset[idx][2] if inertial_exists else None
                    if inertial is not None and isinstance(inertial, torch.Tensor):
                        inertial = inertial.numpy()

                    terrain_label = cluster_to_terrain[new_cluster_labels[idx]]['terrain_label']
                    preference = cluster_to_terrain[new_cluster_labels[idx]]['preference']

                    batch_patches.append(patch)
                    if inertial_exists:
                        batch_inertial.append(inertial)
                    batch_labels.append(terrain_label)
                    batch_prefs.append(preference)

                # Write batch to HDF5
                patches_dset[start_idx:end_idx] = np.stack(batch_patches)
                if inertial_exists:
                    inertial_dset[start_idx:end_idx] = np.stack(batch_inertial)
                terrain_labels_dset[start_idx:end_idx] = batch_labels
                preferences_dset[start_idx:end_idx] = batch_prefs

        print(f"Saved labeled data to: {dataset_save_path}")

        existing_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                existing_config = yaml.safe_load(file) or {}

        existing_config['terrains'] = new_terrains
        with open(config_path, "w") as file:
            yaml.dump(existing_config, file, default_flow_style=None, sort_keys=False)
        print(f"Updated config at: {config_path}")

        success_dialog = Gtk.MessageDialog(
            transient_for=self.parent_window,
            modal=True,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Terrains updated in config.yaml, labeled data saved, and cluster plot generated.",
        )
        success_dialog.show()
        success_dialog.connect("response", lambda dialog, response: dialog.destroy())

def get_children(box):
    return box.get_children()

app = ClusterUI()
exit_status = app.run(sys.argv)
sys.exit(exit_status)
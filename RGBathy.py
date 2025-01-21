import os
import random
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import yaml
import numpy as np
import torch

class RGBathy(Dataset):
    """
    PyTorch Dataset zum Laden von Bildpaaren in verschiedenen Modi (z. B. normal <-> no_waves).
    
    Die Konfiguration wird per Dictionary (oder Datei) übergeben und enthält mindestens:
      - root_dir (str): Hauptordner, in dem sich die Shader-Unterordner befinden.
      - shader_folders (List[str]): Liste der Shader-Unterordner.
      - pair_mode (str): Wie die Bildpaare zusammengestellt werden sollen.
      - split (dict): z.B. {"train": 0.7, "val": 0.15, "test": 0.15}.
      - seed (int): Seed für Reproduzierbarkeit beim Split.
      - proportional_shaders (bool): Ob beim Split prozentual pro Shader gezogen wird oder gemischt.
      - transform_mode (str): "resize" oder "crop".
      - crop_size (int, optional): Größe des Ausschnitts, falls "crop" gewählt.
      - use_augmentation (bool): Falls True, werden einfache Augmentationen angewendet.
    """
    def __init__(self, config, split_mode="train"):
        """
        Args:
            config (dict): Dictionary mit allen notwendigen Parametern.
            split_mode (str): "train", "val" oder "test"
        """
        super().__init__()
        self.config = config
        self.split_mode = split_mode.lower()
        assert self.split_mode in ["train", "val", "test"], "split_mode muss 'train', 'val' oder 'test' sein."

        self.root_dir = config["root_dir"]
        self.shader_folders = sorted(config["shader_folders"])
        self.pair_mode = config["pair_mode"]
        self.split = config["split"]
        self.seed = config["seed"]
        self.proportional_shaders = config.get("proportional_shaders", True)
        self.transform_mode = config.get("transform_mode", "crop").lower()
        self.img_size = config.get("size", 256)
        self.use_augmentation = config.get("use_augmentation", False)

        if self.transform_mode not in ["resize", "crop"]:
            raise ValueError("transform_mode muss entweder 'resize' oder 'crop' sein.")

        # 1) IDs aus den Blacklists und Test-Listen pro Shader laden
        self.shader_blacklists = {}
        self.shader_testsets = {}
        for shader in self.shader_folders:
            shader_path = os.path.join(self.root_dir, shader)
            blacklist_path = os.path.join(shader_path, "blacklist.txt")
            testset_path = os.path.join(shader_path, "test_list.txt")
            self.shader_blacklists[shader] = self._load_list_file(blacklist_path)
            self.shader_testsets[shader] = self._load_list_file(testset_path)

        # 2) Alle Bildnummern (IDs) pro Shader sammeln
        shader_id_dict = self._collect_ids_per_shader()

        # 3) Split in Train/Val/Test
        self.data_ids = self._create_split(shader_id_dict)

        # 4) Erzeuge schließlich eine Liste aller (ShaderFolder, ID) Paare
        self.samples = []
        for shader, ids in self.data_ids[self.split_mode].items():
            for img_id in ids:
                self.samples.append((shader, img_id))

        self.samples.sort()  # Sortiere für Reproduzierbarkeit

        # 5) Vorbereitung der Transformationen
        self.transform = self._build_transforms()

    def _build_transforms(self):
        """
        Erzeugt die Compose-Transformationen für Input und Label.
        Wenn nötig: Resize, Crop, Augmentation.
        """
        transform_list = []
        transform_list.append(T.ToImage())
        if self.transform_mode == "resize":
            transform_list.append(T.Resize((self.img_size, self.img_size)))
        elif self.transform_mode == "crop":
            if self.split_mode == "train":
                transform_list.append(T.RandomCrop((self.img_size, self.img_size)))
            else:
                transform_list.append(T.CenterCrop((self.img_size, self.img_size)))
        else:
            raise ValueError(f"Unbekannter transform_mode: {self.transform_mode}")

        # Optionale Augmentationen
        if self.use_augmentation and self.split_mode == "train":
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
            transform_list.append(T.RandomVerticalFlip(p=0.5))
            # Weitere Augmentationen können hier hinzugefügt werden
            # transform_list.append(T.RandomRotation(10))  # Beispiel: zufällige Drehung

        transform_list.append(T.ToDtype(torch.float32, scale=True)) #(T.ToTensor())

        return T.Compose(transform_list)

    def _load_list_file(self, filepath):
        """
        Lädt eine Datei, in der pro Zeile eine Bildnummer steht.
        Gibt ein Set dieser Bildnummern zurück.
        """
        if not os.path.exists(filepath):
            print(f"[WARNUNG] Datei '{filepath}' nicht gefunden. Rückgabe eines leeren Sets.")
            return set()

        with open(filepath, "r") as f:
            lines = f.readlines()

        ids = set()
        for line in lines:
            line = line.strip()
            # Kommentare oder leere Zeilen ignorieren
            if line.startswith("#") or line == "":
                continue
            ids.add(line.zfill(4))  # Sicherstellen, dass IDs vierstellig sind

        return ids

    def _collect_ids_per_shader(self):
        """
        Durchsucht alle angegebenen Shader-Unterordner nach PNG-Dateien
        und extrahiert die 4-stellige Nummer aus dem Dateinamen.
        Gibt ein Dictionary zurück: { shader_folder: set_of_ids }.
        """
        shader_id_dict = {}
        for shader_folder in self.shader_folders:
            full_path = os.path.join(self.root_dir, shader_folder)
            if not os.path.isdir(full_path):
                print(f"[WARNUNG] Shader-Ordner '{full_path}' ist kein Verzeichnis oder existiert nicht.")
                shader_id_dict[shader_folder] = set()
                continue

            # Suche nach render_*.png
            png_files = glob.glob(os.path.join(full_path, "render_*.png"))
            collected_ids = set()
            for file in png_files:
                filename = os.path.basename(file)
                base, ext = os.path.splitext(filename)  # ('render_0001_no_waves', '.png')
                if base.startswith("render_"):
                    # Extrahiere die ersten vier Ziffern nach 'render_'
                    potential_id = base[7:11]  # '0001'
                    if potential_id.isdigit():
                        # Prüfen, ob blacklisted
                        if potential_id not in self.shader_blacklists.get(shader_folder, set()):
                            collected_ids.add(potential_id)
            shader_id_dict[shader_folder] = collected_ids
        return shader_id_dict

    def _create_split(self, shader_id_dict):
        """
        Teilt die Bild-IDs in Train/Val/Test auf.
        Falls "test_set_file" definiert ist, kommen diese IDs direkt ins Test-Set.
        Danach werden die restlichen IDs prozentual verteilt, wobei
        - train: split["train"]
        - val:   split["val"]
        - test:  split["test"]
        benutzt wird.
        
        Gibt ein Dictionary zurück:
        {
          "train": {shader_folder: set_of_ids},
          "val":   {shader_folder: set_of_ids},
          "test":  {shader_folder: set_of_ids}
        }
        """
        # random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # Container für Resultate
        data_split = {
            "train": {},
            "val": {},
            "test": {}
        }

        for shader, all_ids in shader_id_dict.items():
            # Entferne erst die fixed_test_ids
            test_ids_fixed = all_ids.intersection(self.shader_testsets.get(shader, set()))
            remaining = list(all_ids - self.shader_testsets.get(shader, set()))
            remaining.sort()

            # Durchmischen
            rng.shuffle(remaining)

            # prozentuale Einteilung
            n = len(remaining)
            n_train = int(n * self.split["train"])
            n_val = int(n * self.split["val"])
            # Test = Rest
            train_ids = remaining[:n_train]
            val_ids = remaining[n_train:n_train+n_val]
            test_ids = remaining[n_train+n_val:]

            # Zusammenführen
            data_split["train"][shader] = set(train_ids)
            data_split["val"][shader] = set(val_ids)
            # Hier fügen wir die fixen Test-IDs zusätzlich noch dazu
            data_split["test"][shader] = set(test_ids).union(test_ids_fixed)

            print(f"Shader '{shader}': Train={len(train_ids)}, Val={val_ids}, Test={test_ids}")

        return data_split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Lädt das i-te Paar an Bildern entsprechend des festgelegten pair_mode.
        Gibt (input_tensor, label_tensor) zurück.
        """
        shader_folder, img_id = self.samples[idx]

        # Pfade zu den Bildern ermitteln
        input_path, label_path = self._get_image_paths(shader_folder, img_id)

        # Bilder laden
        try:
            input_img = Image.open(input_path).convert("RGB")
            label_img = Image.open(label_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden der Bilder: {input_path} oder {label_path}. Fehler: {e}")

        # Transformationen anwenden (gemeinsam)
        if self.transform:
            input_img, label_img = self.transform(input_img, label_img)
            # label_img = self.transform(label_img)

        return input_img, label_img

    def _get_image_paths(self, shader_folder, img_id):
        """
        Ermittelt die Pfade zu Input- und Label-Bild basierend auf pair_mode.
        """
        base_dir = os.path.join(self.root_dir, shader_folder)

        # Mögliche Dateinamen:
        #  - normal: "render_XXXX.png"
        #  - ground: "render_XXXX_ground.png"
        #  - no_sunglint: "render_XXXX_no_sunglint.png"
        #  - no_waves: "render_XXXX_no_waves.png"

        normal_file = f"render_{img_id}.png"
        no_sunglint_file = f"render_{img_id}_no_sunglint.png"
        no_waves_file = f"render_{img_id}_no_waves.png"
        ground_file = f"render_{img_id}_ground.png"  # eventuell nicht benötigt

        pair_mode = self.pair_mode.lower()

        # Für die geforderten Modi:
        if pair_mode == "normal_no_waves":
            input_path = os.path.join(base_dir, normal_file)
            label_path = os.path.join(base_dir, no_waves_file)
        elif pair_mode == "normal_no_sunglint":
            input_path = os.path.join(base_dir, normal_file)
            label_path = os.path.join(base_dir, no_sunglint_file)
        elif pair_mode == "no_sunglint_no_waves":
            input_path = os.path.join(base_dir, no_sunglint_file)
            label_path = os.path.join(base_dir, no_waves_file)
        elif pair_mode == "normal_filtered_no_waves":
            # Beispiel: Falls du "normal" nur nimmst, wenn es keinen Sun Glint enthält
            # => Dann brauchst du ggf. vorher eine Filter-Liste.
            # Hier der Vollständigkeit halber nur als Dummy.
            input_path = os.path.join(base_dir, normal_file)
            label_path = os.path.join(base_dir, no_waves_file)
        else:
            raise ValueError(f"Unbekannter pair_mode: {self.pair_mode}")

        # Überprüfen, ob beide Dateien existieren
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input-Bild nicht gefunden: {input_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label-Bild nicht gefunden: {label_path}")

        return input_path, label_path

def load_config(path_to_config):
    """
    Funktion zum Einlesen einer YAML-Konfigurationsdatei.
    """
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 1) Config laden
    config = load_config("config.yaml")

    # 2) Dataset-Objekte für train, val und test erstellen
    train_dataset = RGBathy(config, split_mode="train")
    val_dataset   = RGBathy(config, split_mode="val")
    test_dataset  = RGBathy(config, split_mode="test")

    # 3) DataLoader erstellen (Batching, Shuffling etc.)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["batch_size"],
                                               shuffle=True,  # Shuffle nur bei Training
                                               num_workers=4,  # [Optimierungspunkt] Anzahl der Worker erhöhen
                                               pin_memory=True)  # [Optimierungspunkt] Nutzen bei GPU
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config["batch_size"],
                                             shuffle=False,
                                             num_workers=4,  # [Optimierungspunkt]
                                             pin_memory=True)  # [Optimierungspunkt]
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=False,
                                              num_workers=4,  # [Optimierungspunkt]
                                              pin_memory=True)  # [Optimierungspunkt]

    # 4) Beispielhafter Durchlauf
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Dein Trainingscode...
        # z.B.:
        # model_output = model(inputs)
        # loss = criterion(model_output, labels)
        # ...
        pass

    print("Datasets erfolgreich geladen.")

if __name__ == "__main__":
    main()

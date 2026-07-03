"""AutomatedWindow: batch DINO annotation driven by LLM-suggested prompts."""
import os
import tempfile

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore

from ..config import CUMULATIVE_DIR
from ..optimizer import multi_optimizer, prompt_optimizer
from ..pipeline.dino import load_dino_model, run_image
from .llm import generate_prompts, sort_largest_file
from .style import BTN_BLUE, BTN_GAP, BTN_GREEN, BTN_GREY, btn_qss

class AutomatedWindow(QtWidgets.QWidget):
    def __init__(self, model, processor):
        super().__init__()
        self.generated_prompts = []  # Will store the 5 prompts from LLM
        self.loaded_prompt_file = ""
        self.DINO = load_dino_model(model_size="swint")
        self.model = model
        self.processor = processor
        self.awaiting_keypress = False
        self.init_ui()

    def show_message(self, title, message, level="info"):
        msg_box = QtWidgets.QMessageBox(self)

        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #000000;
            }
            QMessageBox QLabel {
                color: white;
                min-width: 400px;
                padding: 10px;
                font-size: 18px;
                qproperty-alignment: 'AlignLeft | AlignTop';
            }
        """)

        msg_box.setWindowTitle(title)
        msg_box.setText(message)

        # Explicitly remove icon to avoid spacing issues
        msg_box.setIcon(QtWidgets.QMessageBox.NoIcon)

        msg_box.exec_()

    def init_ui(self):
        self.setWindowTitle("Automated Prompt and Confidence Tuning")
        self.showFullScreen()
        self.setStyleSheet("background-color: #454545;")
        screen_h = QtWidgets.QApplication.primaryScreen().geometry().height()
        btn_h = max(40, screen_h // 16)
        font  = max(13, screen_h // 58)
        
        layout = QtWidgets.QVBoxLayout()
        
        back_btn = QtWidgets.QPushButton("Back")
        back_btn.setStyleSheet(btn_qss(BTN_GREY, font))
        back_btn.setFixedSize(200, btn_h)
        back_btn.setToolTip("Return to the main menu.")
        back_btn.clicked.connect(self.go_back)
        layout.addWidget(back_btn, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        # Left layout for folder selection
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_layout.setSpacing(BTN_GAP)

        label_btn = QtWidgets.QPushButton("Select Label Folder")
        label_btn.setStyleSheet(btn_qss(BTN_BLUE, font))
        label_btn.setFixedHeight(btn_h)
        label_btn.setToolTip("Choose the folder of ground-truth .txt labels used as the optimization reference.")
        label_btn.clicked.connect(self.select_label_folder)
        self.left_layout.addWidget(label_btn, alignment=QtCore.Qt.AlignTop)
        
        self.labelled_folder_label = QtWidgets.QLabel("")
        self.labelled_folder_label.setStyleSheet(f"color: white; font-size: {font}px;")
        self.left_layout.addWidget(self.labelled_folder_label, alignment=QtCore.Qt.AlignTop)
        
        unannotated_btn = QtWidgets.QPushButton("Select Image Folder")
        unannotated_btn.setStyleSheet(btn_qss(BTN_BLUE, font))
        unannotated_btn.setFixedHeight(btn_h)
        unannotated_btn.setToolTip("Choose the folder of images to be automatically annotated.")
        unannotated_btn.clicked.connect(self.select_image_folder)
        self.left_layout.addWidget(unannotated_btn, alignment=QtCore.Qt.AlignTop)
        
        self.image_folder_label = QtWidgets.QLabel("")
        self.image_folder_label.setStyleSheet(f"color: white; font-size: {font}px;")
        self.left_layout.addWidget(self.image_folder_label, alignment=QtCore.Qt.AlignTop)
        
        output_btn = QtWidgets.QPushButton("Select Output Folder")
        output_btn.setStyleSheet(btn_qss(BTN_BLUE, font))
        output_btn.setFixedHeight(btn_h)
        output_btn.setToolTip("Choose where the generated boxes/ and segments/ labels are written.")
        output_btn.clicked.connect(self.select_output_folder)
        self.left_layout.addWidget(output_btn, alignment=QtCore.Qt.AlignTop)
        
        self.output_folder_label = QtWidgets.QLabel("")
        self.output_folder_label.setStyleSheet(f"color: white; font-size: {font}px;")
        self.left_layout.addWidget(self.output_folder_label, alignment=QtCore.Qt.AlignTop)
        
        # Right layout for prompt selection
        self.right_layout = QtWidgets.QVBoxLayout()
        self.right_layout.setSpacing(BTN_GAP)
        
        prompt_select_btn = QtWidgets.QPushButton("Prompt Selection")
        prompt_select_btn.setStyleSheet(btn_qss(BTN_BLUE, font))
        prompt_select_btn.setFixedHeight(btn_h)
        prompt_select_btn.setToolTip("Reveal the options for supplying candidate prompts: load a list from "
                                     "file, or generate them with the vision-language model.")
        prompt_select_btn.clicked.connect(self.prompt_selection)
        self.right_layout.addWidget(prompt_select_btn, alignment=QtCore.Qt.AlignTop)
        
        self.bottom_layout = QtWidgets.QVBoxLayout()
        self.bottom_layout.setSpacing(BTN_GAP)
        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setStyleSheet(f"font-size: {font}px; color: white;")
        self.status_label.setAlignment(QtCore.Qt.AlignTop)
        self.status_label.setWordWrap(True)
        self.bottom_layout.addWidget(self.status_label, alignment=QtCore.Qt.AlignCenter)
        start_btn = QtWidgets.QPushButton("Start Annotation")
        start_btn.setStyleSheet(btn_qss(BTN_GREEN, font))
        start_btn.setFixedHeight(btn_h)
        start_btn.setToolTip("Optimize the prompt and confidence against the reference label, then "
                             "annotate every image in the image folder.")
        start_btn.clicked.connect(self.perform_automatic_annotation)
        self.bottom_layout.addWidget(start_btn, alignment=QtCore.Qt.AlignCenter)
        
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.addLayout(self.left_layout)
        main_layout.addLayout(self.right_layout)
        
        layout.addLayout(main_layout)
        layout.addLayout(self.bottom_layout)
        self.setLayout(layout)


    def _undo_annotation(self):
        self.image_label.undo()

    def _redo_annotation(self):
        self.image_label.redo()

    def _save_and_confirm(self):
        if not self.images or not self.output_folder:
            return
        active = self.image_label.get_active_annotations()
        image_path = self.images[self.current_image_index]
        stem = os.path.splitext(os.path.basename(image_path))[0]

        # Write segments file with remaining mask polygons
        seg_dir = self.output_folder + '/segments'
        os.makedirs(seg_dir, exist_ok=True)
        with open(f'{seg_dir}/{stem}.txt', 'w') as f:
            for poly in active:
                coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in poly)
                f.write(f'0 {coords}\n')

        # Write bounding boxes file derived from polygon bounding rects
        box_dir = self.output_folder + '/boxes'
        os.makedirs(box_dir, exist_ok=True)
        with open(f'{box_dir}/{stem}.txt', 'w') as f:
            for poly in active:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                bw = max(xs) - min(xs)
                bh = max(ys) - min(ys)
                f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

        # Re-render image showing only the remaining annotations
        if self.base_cv2_image is not None:
            img = self.base_cv2_image.copy()
            for poly in active:
                pts = np.array(
                    [[int(x * img.shape[1]), int(y * img.shape[0])] for x, y in poly],
                    dtype=np.int32
                )
                cv2.drawContours(img, [pts], -1, (255, 0, 255), 2)
            self.show_result_image(img)

        # Exit edit mode
        self.edit_btn.setChecked(False)

        msg = QtWidgets.QMessageBox()
        msg.setStyleSheet("QLabel { color: black; } QMessageBox { background-color: white; }")
        msg.setText(f"Saved {len(active)} annotation(s) for {stem}.")
        msg.exec_()

    def go_back(self):
        from .splash import MainWindow
        self.main_window = MainWindow(self.model, self.processor)
        self.main_window.show()
        self.close()

    def select_image_folder(self):
        options = QtWidgets.QFileDialog.Options()
        dialog = QtWidgets.QFileDialog(self, "Select Image Folder", CUMULATIVE_DIR, options=options)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dialog.setStyleSheet("QWidget { background-color: white; color: black; }")
        dialog.setOption(QtWidgets.QFileDialog.ReadOnly, True)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            folder = dialog.selectedFiles()[0]
            if folder:
                # Load images from the selected folder.
                self.images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                self.image_folder = folder
                self.image_folder_label.setText(f"Image Folder: {self.image_folder}")


    def select_output_folder(self):
        options = QtWidgets.QFileDialog.Options()
        # Remove the ReadOnly option
        # options |= QtWidgets.QFileDialog.ReadOnly
        # Optionally, remove the DontUseNativeDialog option to use the native file dialog
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog

        dialog = QtWidgets.QFileDialog(self, "Select Output Folder", options=options)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dialog.setStyleSheet("QWidget { background-color: white; color: black; }")

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.output_folder = dialog.selectedFiles()[0]
            if self.output_folder:
                # You can add any additional logic needed when the output folder is selected
                self.output_folder_label.setText(f"Output Folder: {self.output_folder}")

    def select_label_folder(self):
        options = QtWidgets.QFileDialog.Options()
        dialog = QtWidgets.QFileDialog(self, "Select Label Folder", options=options)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QtWidgets.QFileDialog.ReadOnly, True)
        dialog.setStyleSheet("QWidget { background-color: white; color: black; }")

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            folder = dialog.selectedFiles()[0]
            if folder:
                # Load label files from the selected folder
                self.label_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.txt')]
                if self.label_files:
                    self.label_folder = folder
                    self.labelled_folder_label.setText(f"Label Folder: {self.label_folder}")
                    # Optionally sort files or trigger further processing here
                else:
                    # Notify the user if the folder is empty or has no label files
                    message_box = QtWidgets.QMessageBox()
                    message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
                    message_box.setText("The selected folder does not contain any .txt label files.")
                    message_box.exec_()

    def update_status(self, text):
        self.status_label.setText(text)
        QtWidgets.QApplication.processEvents()

    def perform_automatic_annotation(self):
        if not hasattr(self, 'label_folder') or not hasattr(self, 'image_folder') or not hasattr(self, 'output_folder'):
            self.show_message("Error", "Please select Label, Image, and Output folders before starting.")
            return
        self.status_label.setText("Sorting label files...")
        QtWidgets.QApplication.processEvents()

        sorted_txt_files = sort_largest_file(self.label_folder)
        reference_txt = os.path.join(self.label_folder, sorted_txt_files[0])
        _base_name = sorted_txt_files[0].split(".txt")[0]
        reference_image = None
        for _ext in ('.jpg', '.jpeg', '.png'):
            _candidate = os.path.join(self.image_folder, _base_name + _ext)
            if os.path.exists(_candidate):
                reference_image = _candidate
                break
        if not reference_image:
            self.show_message("Error", f"Could not find matching image for label '{_base_name}'. Check that image and label filenames match.")
            return

        if self.generated_prompts:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmp:
                for prompt in self.generated_prompts:
                    tmp.write(prompt + "\n")
                tmp_path = tmp.name
        elif self.loaded_prompt_file:
            tmp_path = self.loaded_prompt_file
        else:
            self.show_message("Error", "No prompts loaded or generated. Please select or generate prompts.", level="error")
            return

        self.status_label.setText("Optimizing prompts for best IoU...")
        QtWidgets.QApplication.processEvents()
        prompt_result = prompt_optimizer(
            prompts_file=tmp_path,
            gt_path=reference_txt,
            img_path=reference_image,
            save_file="best.txt",
            threshold=0.8,
            DINO=self.DINO
        )

        top2 = [result[0] for result in prompt_result][:2]

        self.status_label.setText("Refining confidence scores...")
        QtWidgets.QApplication.processEvents()
        best_prompt, best_conf = multi_optimizer(img_dir=reference_image, gt_label_dir=reference_txt, DINO=self.DINO, prompts=top2, threshold=0.8, callback=lambda prompt, i, total: self.update_status(f"Confidence tuning: '{prompt}' ({i+1}/{total})")
        )
        self.update_status(
            f"The best prompt is:\n{best_prompt}\n\n"
            f"The best confidence score is:\n{best_conf:.2f}"
        )

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Continue")
        msg_box.setText("Press Enter to continue...")
        msg_box.setStyleSheet("QLabel{font-size: 24px; color: white;} QMessageBox{background-color: black;}")
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec_()

        self.status_label.setText("Starting labelling of images...")
        QtWidgets.QApplication.processEvents()

        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(image_files)

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(self.image_folder, image_file)
            run_image(self.DINO, img_dir=image_path, output_dir=self.output_folder+"/segments", prompt=best_prompt, conf=best_conf, box_threshold=0.8, save_dir=self.output_folder+"/boxes")
            self.status_label.setText(f"Labelling image {i + 1} of {total_images}")
            QtWidgets.QApplication.processEvents()

        self.status_label.setText(f"LABELLING COMPLETE\nOutput saved to: {self.output_folder}")


    def prompt_selection(self):
        if not hasattr(self, "prompt_buttons_added"):
            screen_h = QtWidgets.QApplication.primaryScreen().geometry().height()
            btn_h = max(40, screen_h // 16)
            font  = max(13, screen_h // 58)
            list_prompts_btn = QtWidgets.QPushButton("List of Prompts")
            list_prompts_btn.setStyleSheet(btn_qss(BTN_BLUE, font))
            list_prompts_btn.setFixedHeight(btn_h)
            list_prompts_btn.setToolTip("Load candidate prompts from a .txt or .csv file (one per line).")
            list_prompts_btn.clicked.connect(self.handle_list_of_prompts)
            self.right_layout.addWidget(list_prompts_btn, alignment=QtCore.Qt.AlignTop)

            generate_prompts_btn = QtWidgets.QPushButton("Generate Prompts")
            generate_prompts_btn.setStyleSheet(btn_qss(BTN_BLUE, font))
            generate_prompts_btn.setFixedHeight(btn_h)
            generate_prompts_btn.setToolTip("Use the vision-language model to propose candidate prompts from a sample image.")
            generate_prompts_btn.clicked.connect(self.handle_generate_prompts)
            self.right_layout.addWidget(generate_prompts_btn, alignment=QtCore.Qt.AlignTop)

            self.prompt_buttons_added = True

    def handle_list_of_prompts(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        dialog = QtWidgets.QFileDialog(self, "Select Prompts List", "", "Text Files (*.txt);;CSV Files (*.csv)", options=options)
        dialog.setStyleSheet("QWidget { background-color: white; color: black; }")
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.loaded_prompt_file = dialog.selectedFiles()[0]
            self.generated_prompts = []
            try:
                with open(self.loaded_prompt_file, "r", encoding="utf-8") as f:
                    loaded = [line.strip() for line in f if line.strip()]
                self.show_message("Loaded Prompts", "\n".join(loaded[:10]))  # Preview top 10
            except Exception as e:
                self.show_message("Error Reading File", str(e), level="error")

    def handle_generate_prompts(self):
        options = QtWidgets.QFileDialog.Options()
        dialog = QtWidgets.QFileDialog(self, "Select Sample Image", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        dialog.setStyleSheet("QWidget { background-color: white; color: black; }")

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            sample_image_path = dialog.selectedFiles()[0]

            if sample_image_path:
                # Explicitly create QInputDialog to style the input box for real-time visibility
                input_dialog = QtWidgets.QInputDialog(self)
                input_dialog.setStyleSheet("""
                    QLabel {
                        font-size: 24px;  /* Adjust the prompt text font size */
                        color: white;
                    }
                    QLineEdit {
                        font-size: 24px;  /* Adjust the user input text font size */
                        color: white;
                        background-color: #000000;
                    }
                    QInputDialog {
                        background-color: #000000;
                        selection-background-color: #3E3E42;
                    }
                """)

                input_dialog.setWindowTitle("Object to Describe")
                input_dialog.setLabelText("Enter the object in the image:")

                # Execute the input dialog explicitly
                if input_dialog.exec_() == QtWidgets.QDialog.Accepted:
                    manual_entry = input_dialog.textValue()

                    if manual_entry:
                        prompts = generate_prompts(sample_image_path, manual_entry, self.model, self.processor)

                        if prompts:
                            self.generated_prompts = prompts
                            self.loaded_prompt_file = ""
                            self.show_message("Generated Prompts", "\n".join(prompts))
                        else:
                            self.show_message("Prompt Generation Failed", "No prompts were returned.", level="warn")

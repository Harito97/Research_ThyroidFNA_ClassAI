import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import seaborn as sns
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image as ReportLabImage
import joblib

class DataExplorer:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.dataset_versions = self.config['dataset_versions']
        self.dataset_paths = self.config['dataset_paths']
        self.output_dir = self.config['output_dir']
        self.max_images_per_class = self.config['max_images_per_class']
        
        os.makedirs(self.output_dir, exist_ok=True)

    def count_images(self, dataset_path):
        class_counts = {'train': {}, 'valid': {}}
        for split in ['train', 'valid']:
            split_path = os.path.join(dataset_path, split)
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    class_counts[split][class_name] = len(os.listdir(class_path))
        return class_counts

    def visualize_class_distribution(self, class_counts, dataset_name):
        plt.figure(figsize=(12, 6))
        df = pd.DataFrame(class_counts)
        df.plot(kind='bar')
        plt.title(f'Class Distribution in {dataset_name}')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.legend(['Train', 'Valid'])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{dataset_name}_class_distribution.png'))
        plt.close()

    def load_images(self, dataset_path):
        images = []
        labels = []
        for split in ['train', 'valid']:
            split_path = os.path.join(dataset_path, split)
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path)[:self.max_images_per_class//6]:
                        img_path = os.path.join(class_path, img_name)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((64, 64))  # Resize for consistency
                        images.append(np.array(img).flatten())
                        labels.append(class_name)
        return np.array(images), np.array(labels)

    def visualize_tsne(self, images, labels, dataset_name):
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(images)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=pd.Categorical(labels).codes, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f't-SNE Visualization of {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{dataset_name}_tsne.png'))
        plt.close()
        
        return reduced_data

    def create_pdf_report(self):
        doc = SimpleDocTemplate(os.path.join(self.output_dir, "dataset_analysis_report.pdf"), pagesize=letter)
        story = []
        
        for dataset_name in self.dataset_versions:
            story.append(ReportLabImage(os.path.join(self.output_dir, f'{dataset_name}_class_distribution.png'), width=500, height=300))
            story.append(ReportLabImage(os.path.join(self.output_dir, f'{dataset_name}_tsne.png'), width=500, height=300))
        
        doc.build(story)

    def analyze_dataset(self, dataset_path, dataset_name):
        print(f"Analyzing {dataset_name}...")
        
        # Count images and visualize distribution
        class_counts = self.count_images(dataset_path)
        self.visualize_class_distribution(class_counts, dataset_name)
        
        # Load images and perform t-SNE
        images, labels = self.load_images(dataset_path)
        reduced_data = self.visualize_tsne(images, labels, dataset_name)
        
        # Save reduced data
        joblib.dump(reduced_data, os.path.join(self.output_dir, f'{dataset_name}_reduced_data.joblib'))

    def run(self):
        for dataset_name, dataset_path in self.dataset_paths.items():
            self.analyze_dataset(dataset_path, dataset_name)

        self.create_pdf_report()
        print("Analysis complete. PDF report generated.")

def run(config_path):
    explorer = DataExplorer(config_path)
    explorer.run()
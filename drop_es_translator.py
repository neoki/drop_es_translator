import json
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, item = self.data[idx]
        return key, item['passage'], item['qa_pairs']

def collate_fn(batch):
    keys, passages, qa_pairs = zip(*batch)
    return keys, passages, qa_pairs

def translate_batch(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

def translate_dataset(data, input_file, output_file, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU model: {torch.cuda.get_device_name(0)}")
        logging.info(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logging.info(f"Memoria GPU usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logging.info(f"Memoria GPU libre: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        logging.info("Running on CPU")

    model_name = "Helsinki-NLP/opus-mt-en-es"
    logging.info(f"Loading model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    
    if device.type == 'cuda':
        model = model.half()  # Use FP16 if on GPU
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

    logging.info("Iniciando la creación del dataset")
    dataset = TranslationDataset(data)
    logging.info(f"Dataset creado con {len(dataset)} elementos")

    logging.info(f"Creando DataLoader con batch_size={batch_size}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    logging.info("DataLoader creado")

    translated_data = {}

    logging.info("Comenzando el proceso de traducción")
    for batch in tqdm(dataloader, desc="Translating"):
        keys, passages, qa_pairs_batch = batch
        
        try:
            translated_passages = translate_batch(passages, model, tokenizer, device)
            logging.info(f"Pasajes traducidos: {len(translated_passages)}")
        except Exception as e:
            logging.error(f"Error al traducir pasajes: {str(e)}")
            raise

        for key, translated_passage, qa_pairs in zip(keys, translated_passages, qa_pairs_batch):
            translated_item = data[key].copy()
            translated_item['passage'] = translated_passage
            
            questions = [qa['question'] for qa in qa_pairs]
            translated_questions = translate_batch(questions, model, tokenizer, device)
            
            for qa, translated_question in zip(translated_item['qa_pairs'], translated_questions):
                qa['question'] = translated_question
                if qa['answer']['spans']:
                    qa['answer']['spans'] = translate_batch(qa['answer']['spans'], model, tokenizer, device)
            
            translated_data[key] = translated_item

        if len(translated_data) % 100 == 0:
            logging.info(f"Guardando progreso: {len(translated_data)} elementos traducidos")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)

        # Liberar memoria CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logging.info("Proceso de traducción completado")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

def main():
    input_file = 'drop_dataset_dev.json'
    input_name = os.path.splitext(input_file)[0]
    output_file = f'{input_name}_GPU_Optimized_ES.json'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Archivo {input_file} cargado correctamente")
    except Exception as e:
        logging.error(f"Error al cargar el archivo {input_file}: {str(e)}")
        return

    try:
        translate_dataset(data, input_file, output_file)
    except Exception as e:
        logging.error(f"Error durante la traducción: {str(e)}")
        logging.error(f"Detalles del error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
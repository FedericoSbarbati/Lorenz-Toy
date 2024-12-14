import json
import csv

def json_to_csv(json_file, csv_file):
    # Leggi il file JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Estrai le intestazioni delle colonne
    headers = set()
    for key, value in data.items():
        headers.update(value.keys())
    headers = list(headers)

    # Scrivi il file CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model'] + headers)  # Aggiungi l'intestazione

        for model_name, model_data in data.items():
            row = [model_name]
            for header in headers:
                value = model_data.get(header, "")
                # Converte le liste in stringhe
                if isinstance(value, list):
                    value = ",".join(map(str, value))
                row.append(value)
            writer.writerow(row)

    print(f"JSON converted to CSV: {csv_file}")

def csv_to_json(csv_file, json_file):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = {}

        for row in reader:
            model_name = row['Model']
            model_data = {key: value for key, value in row.items() if key != 'Model'}

            # Converte stringhe di liste in liste
            if 'decoder_layers' in model_data and model_data['decoder_layers']:
                model_data['decoder_layers'] = list(map(int, model_data['decoder_layers'].split(',')))

            # Converte altri campi numerici se necessario
            for key, value in model_data.items():
                if key != 'decoder_layers' and value.isdigit():
                    model_data[key] = int(value)
                elif key != 'decoder_layers' and value.replace('.', '', 1).isdigit():
                    model_data[key] = float(value)

            data[model_name] = model_data

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"CSV converted back to JSON: {json_file}")


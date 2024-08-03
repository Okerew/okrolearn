from okrolearn.dataloader import DataLoader
# Load JSON data
json_tensor = DataLoader.load_json('data.json')
print(json_tensor)
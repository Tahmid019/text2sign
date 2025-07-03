import os


for filename in os.listdir('.'):
    if filename.endswith('.jpg'):
        base = os.path.splitext(filename)[0]
        new_filename = base + '.gif'
        os.rename(filename, new_filename)
        print(f"Renamed: {filename} -> {new_filename}")

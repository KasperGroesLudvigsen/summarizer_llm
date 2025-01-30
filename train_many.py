import train
import pathlib
import os 
import yaml

if __name__ == "__main__":

    config_dir = "training_configs"

    config_files = os.listdir(config_dir)
    
    config_files = [f for f in config_files if pathlib.Path(f).suffix == ".yml"]

    configs = []

    for file in config_files:
        config_path = f"{config_dir}/{file}"
        # Load baseline configuration
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            config["config_path"] = config_path
            
        configs.append(config)

    print(configs)

    for config in configs:

        try:

            train.train(config)

        except Exception as e:

            print(f"Could not fine tune model due to error:\n {e}\n")


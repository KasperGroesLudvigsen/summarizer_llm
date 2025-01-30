import train
import pathlib
import os 
import yaml

if __name__ == "__main__":

    config_path = "training_configs/config6.yml"
    # Load baseline configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config["config_path"] = config_path
            
    try:

        train.train(config)

    except Exception as e:

        print(f"Could not fine tune model due to error:\n {e}\n")


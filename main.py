""" 
Main file that initialize the docker image and invoke the system analysis and the final platform hosting.
"""
import os, sys
import subprocess

from analytics.FeatEng import FeaturesEngineering

# Value is True/False for development/deployment
TRAIN_MODEL = True

if __name__ == "__main__":
    #### Start the database docker
    output = subprocess.run(["sudo","docker", "start" ,"uts-database"], stdout=subprocess.PIPE)
    if output.returncode != 0:
        print(output.stdout.decode("utf-8"))
        sys.exit("Failed to launch database docker!")
    
    #### Generate the required features
    ft = FeaturesEngineering()
    print(ft.get_active_player_table().columns)

    #### Analyze and clean
    
    if TRAIN_MODEL:
        #### Run model training
        pass

    #### Initialize the interface

    
import os
import shutil
import pandas as pd

'''
Update the Config.toml with data from the participant data file
'''
def update_participant_config(config_path, height, mass):
    if not os.path.exists(config_path):
        print(f"Config file does not exist at {config_path}.")
        return
    
    # Read the current file
    with open(config_path, 'r') as file:
        lines = file.readlines()
    
    # Modify the lines with the new values
    updated_lines = []
    for line in lines:
        if line.strip().startswith('participant_height'):
            updated_lines.append(f'participant_height = {height}\n')
        elif line.strip().startswith('participant_mass'):
            updated_lines.append(f'participant_mass = {mass}\n')
        else:
            updated_lines.append(line)
    
    # Write the updated lines back to the file
    with open(config_path, 'w') as file:
        file.writelines(updated_lines)
    
    print(f"Updated participant config: {config_path}")

'''
MAIN
'''
def mk_data_struct():
    # Get the setup data directory and participant data file paths
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    setup_data_dir = os.path.join(script_dir, 'Setup_data')
    excel_file_path = os.path.join(cwd, 'Participant_Data.xlsx')

    # Copy Batch Config.toml file
    batch_config_file = os.path.join(setup_data_dir, 'Batch_Config.toml')
    dest_batch_config_file = os.path.join(cwd, 'Config.toml')

    if os.path.isfile(batch_config_file):
        print(f"Copying {batch_config_file} to {dest_batch_config_file}...")
        shutil.copy2(batch_config_file, dest_batch_config_file)
    else:
        print(f"Batch Config file does not exist at {batch_config_file}. Exiting.")
        return
    
    try: 
        # Load the participant data from the Excel file
        participant_data = pd.read_excel(excel_file_path)

        # Process data for every participant
        for _, row in participant_data.iterrows():
            participant_id = str(row['Participant ID'])
            height = row['Height']
            mass = row['Weight']

            # Find participant directories starting with the participant ID
            participant_dirs = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d)) and d.startswith(participant_id + "-")]

            if not participant_dirs:
                print(f"No participant directories found for ID {participant_id}. Skipping.")
                continue

            for participant_dir in participant_dirs:
                participant_dir_path = os.path.join(cwd, participant_dir)
                print(f"Processing Participant {participant_id} in directory {participant_dir_path}...")

                # Copy Config.toml files
                trial_config_file = os.path.join(setup_data_dir, 'Trial_Config.toml')
                dest_trial_config_file = os.path.join(participant_dir_path, 'Config.toml')

                if os.path.isfile(trial_config_file):
                    print(f"Copying {trial_config_file} to {dest_trial_config_file}...")
                    shutil.copy2(trial_config_file, dest_trial_config_file)
                    
                    # Update Config.toml with participant height, mass
                    update_participant_config(dest_trial_config_file, height, mass)

                else:
                    print(f"Trial Config file does not exist at {trial_config_file}. Skipping.")
                    continue
    
    except Exception as e:
        print(f"Error: {e}")
        print("No Participant_Data.xlsx file found. Only Batch Config was generated.")
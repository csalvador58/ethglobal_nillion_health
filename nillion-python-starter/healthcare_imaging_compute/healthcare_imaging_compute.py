import asyncio
import time
import numpy as np
import pandas as pd
from utils.func import (
    calc_scaling_factor,
    compute_scaled_data,
    plot_distributions,
    compute_prediction,
)
import os
import py_nillion_client as nillion
import sys
from dotenv import load_dotenv
import pprint
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers.nillion_client_helper import create_nillion_client
from helpers.nillion_keypath_helper import getUserKeyFromFile, getNodeKeyFromFile

load_dotenv()


async def main():
    print("\n\n******* Healthcare Imaging Compute Program *******\n\n")

    # Create a parser
    parser = argparse.ArgumentParser(description="Check if plot is disabled")
    parser.add_argument("--disable_plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    # Load dataset
    script_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(script_directory, "data", "cancer_data.csv")
    health_data = pd.read_csv(csv_file_path)

    # Replace values in the 'Diagnosis' column
    health_data["Diagnosis"] = health_data["Diagnosis"].apply(
        lambda x: 1 if x == "M" else 0
    )

    # Print the rows of data
    print("\nFirst 50 rows of data' column:")
    print(health_data.head(50))

    # Remove ID column
    health_data = health_data.drop("ID", axis=1)

    # Split Dataset, 80 percent training, 20 percent testing
    train_size = int(0.8 * len(health_data))
    train_data = health_data[:train_size]
    test_data = health_data[train_size:]

    # Create a subset that is 25% of train_data
    subset_25_size = int(0.25 * len(train_data))
    subset_75_size = int(0.75 * len(train_data))
    subset_sm_data = train_data[:subset_25_size]

    # Create a subset that is 75% of train_data
    subset_lg_data = train_data[subset_25_size:]

    if not args.disable_plot:
        # Plot datasets
        plot_distributions(train_data, 7, 2, 25, "Distribution Plots of Training Data")
        plot_distributions(
            subset_sm_data, 7, 2, 25, "Distribution Plots of Subset - Sm"
        )
        plot_distributions(
            subset_lg_data, 7, 2, 25, "Distribution Plots of Subset - Lg"
        )

    # Set up data for training
    X_train = train_data.drop("Diagnosis", axis=1).values
    X_train_subset_sm = subset_sm_data.drop("Diagnosis", axis=1).values
    X_train_subset_lg = subset_lg_data.drop("Diagnosis", axis=1).values
    y_train = train_data["Diagnosis"].values
    y_train_subset_sm = subset_sm_data["Diagnosis"].values
    y_train_subset_lg = subset_lg_data["Diagnosis"].values

    # Select a random test_data and drop 'Diagnosis' column
    random_row = test_data.sample(n=1)
    # random_row = test_data.sample(n=1, random_state=20) # fixed random_state=20, selects 541st row with Diagnosis=0
    print("\nTest instance from test_data before removing target field:")
    print(random_row)
    X_test = random_row.drop("Diagnosis", axis=1).values

    # Add a column of ones to X_train for the intercept term
    X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_train_subset_sm_augmented = np.hstack(
        (np.ones((X_train_subset_sm.shape[0], 1)), X_train_subset_sm)
    )
    X_train_subset_lg_augmented = np.hstack(
        (np.ones((X_train_subset_lg.shape[0], 1)), X_train_subset_lg)
    )

    # Calculate coefficients using the normal equation
    theta = (
        np.linalg.inv(X_train_augmented.T @ X_train_augmented)
        @ X_train_augmented.T
        @ y_train
    )
    theta_subset_sm = (
        np.linalg.inv(X_train_subset_sm_augmented.T @ X_train_subset_sm_augmented)
        @ X_train_subset_sm_augmented.T
        @ y_train_subset_sm
    )
    theta_subset_lg = (
        np.linalg.inv(X_train_subset_lg_augmented.T @ X_train_subset_lg_augmented)
        @ X_train_subset_lg_augmented.T
        @ y_train_subset_lg
    )

    # Coefficients
    print("\n\nComputed theta values:")
    print("\nTheta (full train data):")
    print(theta)
    print("\nTheta (theta_subset_sm):")
    print(theta_subset_sm)
    print("\nTheta (theta_subset_lg):")
    print(theta_subset_lg)

    # Determine best scaling factor and compute scaled theta values
    precision = 20
    factor_theta = calc_scaling_factor(theta, precision)
    factor_theta_subset_sm = calc_scaling_factor(theta_subset_sm, precision)
    factor_theta_subset_lg = calc_scaling_factor(theta_subset_lg, precision)
    scaling_factor = max(factor_theta, factor_theta_subset_sm, factor_theta_subset_lg)

    scaled_theta = compute_scaled_data(theta, scaling_factor)
    scaled_theta_subset_sm = compute_scaled_data(theta_subset_sm, scaling_factor)
    scaled_theta_subset_lg = compute_scaled_data(theta_subset_lg, scaling_factor)

    print("\nScaling Factor:")
    print(scaling_factor)
    print("\nScaled Coefficients (theta):")
    print(scaled_theta)
    print("\nScaled Coefficients (scaled_theta_subset_sm):")
    print(scaled_theta_subset_sm)
    print("\nScaled Coefficients (scaled_theta_subset_lg):")
    print(scaled_theta_subset_lg)

    # Predictions for the test data
    X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    print("\nX_test_augmented:")
    print(X_test_augmented)

    # prediction = X_test_augmented @ theta
    # prediction_subset_sm = X_test_augmented @ theta_subset_sm
    # prediction_subset_lg = X_test_augmented @ theta_subset_lg

    # print("\nPrediction for the full test data using np:")
    # print(prediction)

    # print("\nPrediction for the test data using the small subset of train data:")
    # print(prediction_subset_sm)
    # print("\nPrediction for the test data using the large subset of train data:")
    # print(prediction_subset_lg)

    # Compute prediction value using combined thetas from subsets
    # Calculate weights for each subset
    weights_subset_sm = subset_25_size / train_size
    weights_subset_lg = subset_75_size / train_size

    # Use weighted average to combine thetas
    combined_theta = (theta_subset_sm * weights_subset_sm) + (
        theta_subset_lg * weights_subset_lg
    )
    print("\nCombined theta via weight average of train datasets:")
    print(combined_theta)

    prediction_subsets_combined = X_test_augmented @ combined_theta
    print("\nPrediction for the test data using the combined theta:")
    print(prediction_subsets_combined)

    # Setup final computation without use of np
    # Get the values from the augmented test data row
    X_test_values = X_test_augmented[0]

    # Scale the test data
    scaled_test_data = compute_scaled_data(X_test_values, scaling_factor)
    print("\nScaled Test Data:")
    print(scaled_test_data)

    # Perform element-wise multiplication and summation on full test data
    Y_prediction = compute_prediction(X_test_values, theta)

    print("\nY_Prediction value and classification on the full test data:")
    print(f"{Y_prediction}, classifying as {'M' if round(Y_prediction) == 1 else 'B'}")

    # Perform element-wise multiplication and summation on scaled data
    # Descale test and theta values
    descaled_test_data = [value / scaling_factor for value in scaled_test_data]
    descaled_theta = [value / scaling_factor for value in scaled_theta]

    Y_prediction_scaled = compute_prediction(descaled_test_data, descaled_theta)
    
    print(
        "\n**Scaled Values check** Y_Prediction value and classification on the full test data (w/scaled values), result expected to be identical:"
    )
    print(
        f"{Y_prediction_scaled}, classifying as {'M' if round(Y_prediction) == 1 else 'B'}"
    )

    # Perform element-wise multiplication and summation on subset_sm data
    Y_prediction_subset_sm = compute_prediction(X_test_values, theta_subset_sm)
    
    print("\nY_Prediction value and classification on the subset_sm test data:")
    print(
        f"{Y_prediction_subset_sm}, classifying as {'M' if round(Y_prediction) == 1 else 'B'}"
    )

    # Perform element-wise multiplication and summation on subset_lg data
    Y_prediction_subset_lg = compute_prediction(X_test_values, theta_subset_lg)

    print("\nY_Prediction value and classification on the subset_lg test data:")
    print(
        f"{Y_prediction_subset_lg}, classifying as {'M' if round(Y_prediction) == 1 else 'B'}"
    )

    #### Nillion program

    # Setup nillion
    print(os.getenv("NILLION_USERKEY_PATH_PARTY_1"))
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    userkey = getUserKeyFromFile(os.getenv("NILLION_USERKEY_PATH_PARTY_1"))

    nodekey = getNodeKeyFromFile(os.getenv("NILLION_NODEKEY_PATH_PARTY_1"))
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id()
    user_id = client.user_id()
    patient_name = "Patient_test_img"
    health_pro_1_name = "HP_1"
    health_pro_2_name = "HP_2"
    program_name = "healthcare_imaging_compute"
    program_mir_path = f"../programs-compiled/{program_name}.nada.bin"

    # store program
    action_id = await client.store_program(cluster_id, program_name, program_mir_path)
    program_id = f"{user_id}/{program_name}"
    print("Stored program. action_id:", action_id)
    print("Stored program_id:", program_id)

    exit(1)
    # Create Secrets

    # Setup Nillion parties
    party_patient_dict = {}
    party_hp_1_dict = {}
    party_hp_2_dict = {}

    # Add secret integers to parties
    party_hp_1_dict["hp_1_data_size"] = nillion.SecretInteger(len(X_train_subset_sm))
    party_hp_2_dict["hp_2_data_size"] = nillion.SecretInteger(len(X_train_subset_lg))

    for i in range(train_data.columns.size):
        party_patient_dict[f"patient_image_data_{i}"] = nillion.SecretInteger(
            scaled_test_data[i]
        )
        party_hp_1_dict[f"hp_1_param_{i}"] = nillion.SecretInteger(
            scaled_theta_subset_sm[i]
        )
        party_hp_2_dict[f"hp_2_param_{i}"] = nillion.SecretInteger(
            scaled_theta_subset_lg[i]
        )

    print("Party Patient:")
    pprint.PrettyPrinter(indent=4).pprint(party_patient_dict)
    print("Party HP 1:")
    pprint.PrettyPrinter(indent=4).pprint(party_hp_1_dict)
    print("Party HP 2:")
    pprint.PrettyPrinter(indent=4).pprint(party_hp_2_dict)

    # Parties store secrets
    party_hp_1_secrets = nillion.Secrets(party_hp_1_dict)
    party_hp_2_secrets = nillion.Secrets(party_hp_2_dict)

    # Bind party secrets to program
    secret_bindings = nillion.ProgramBindings(program_id)
    secret_bindings.add_input_party(health_pro_1_name, party_id)
    secret_bindings.add_input_party(health_pro_2_name, party_id)

    # Give permissions
    secret_permissions = nillion.Permissions.default_for_user(user_id)
    secret_permissions.add_compute_permission({user_id: {program_id}})

    # Store in the network and retrieve store Ids
    store_ids = []

    print(f"Storing secrets for Party HP 1: {party_hp_1_secrets}")
    store_id = await client.store_secrets(
        cluster_id, secret_bindings, party_hp_1_secrets, secret_permissions
    )
    store_ids.append(store_id)
    print(f"Stored Party HP 1 with store_id: {store_id}")

    print(f"Storing secrets for Party HP 2: {party_hp_2_secrets}")
    store_id = await client.store_secrets(
        cluster_id, secret_bindings, party_hp_2_secrets, secret_permissions
    )
    store_ids.append(store_id)
    print(f"Stored Party HP 2 with store_id: {store_id}")

    # Bind the parties in the computation to the client to set input and output parties
    compute_bindings = nillion.ProgramBindings(program_id)
    compute_bindings.add_input_party(health_pro_1_name, party_id)
    compute_bindings.add_input_party(health_pro_2_name, party_id)
    compute_bindings.add_output_party(patient_name, party_id)
    computation_time_secrets = nillion.Secrets(party_patient_dict)

    # Setup public variables
    public_variables = {"prediction_init": 0}

    # Compute the prediction on the secret data
    compute_id = await client.compute(
        cluster_id,
        compute_bindings,
        store_ids,
        computation_time_secrets,
        secret_permissions,
        nillion.PublicVariables(public_variables),
    )
    print(f"The computation was sent to the network - compute_id: {compute_id}")

    # Compute Results
    compute_event_result = await client.next_compute_event()
    while not isinstance(compute_event_result, nillion.ComputeFinishedEvent):
        time.sleep(0.5)
        compute_event_result = await client.next_compute_event()

    print(f"✅  Compute complete for compute_id {compute_event_result.uuid}")
    print(f"🖥️  The returned value: {compute_event_result.result.value}")
    print(f"Scaling Factor: {scaling_factor}")
    patient_prediction = (
        compute_event_result.result.value["patient_test_prediction"] / scaling_factor
    )
    print(f"🔮  The prediction is {patient_prediction}")


if __name__ == "__main__":
    asyncio.run(main())

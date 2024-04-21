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
from config import (
    CONFIG_PROGRAM_NAME,
    CONFIG_TEST_PARTY_1,
    CONFIG_HEALTH_PROVIDER_PARTIES,
)

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
    precision = 10
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
    
    #############################################
    ############# Nillion section ###############
    #############################################

    print("\n\n******* Nillion Program *******\n\n")

    # Setup nillion
    print("\nSetting up clients...\n")
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    program_mir_path = f"../programs-compiled/{CONFIG_PROGRAM_NAME}.nada.bin"

    # Setup Parties
    client_test_patient = create_nillion_client(
        getUserKeyFromFile(CONFIG_TEST_PARTY_1["userkey_file"]),
        getNodeKeyFromFile(CONFIG_TEST_PARTY_1["nodekey_file"]),
    )
    client_health_provider_1 = create_nillion_client(
        getUserKeyFromFile(CONFIG_HEALTH_PROVIDER_PARTIES[0]["userkey_file"]),
        getNodeKeyFromFile(CONFIG_HEALTH_PROVIDER_PARTIES[0]["nodekey_file"]),
    )
    client_health_provider_2 = create_nillion_client(
        getUserKeyFromFile(CONFIG_HEALTH_PROVIDER_PARTIES[1]["userkey_file"]),
        getNodeKeyFromFile(CONFIG_HEALTH_PROVIDER_PARTIES[1]["nodekey_file"]),
    )

    # Get party and user IDs
    print("\nSetting up Party and User IDs...\n")
    party_id_test_patient = client_test_patient.party_id()
    user_id_test_patient = client_test_patient.user_id()
    
    party_id_health_provider_1 = client_health_provider_1.party_id()
    user_id_health_provider_1 = client_health_provider_1.user_id()
    
    party_id_health_provider_2 = client_health_provider_2.party_id()
    user_id_health_provider_2 = client_health_provider_2.user_id()
    
    print("Party ID Test Patient:", party_id_test_patient)
    print("User ID Test Patient:", user_id_test_patient)
    print("Party ID Health Provider 1:", party_id_health_provider_1)
    print("User ID Health Provider 1:", user_id_health_provider_1)
    print("Party ID Health Provider 2:", party_id_health_provider_2)
    print("User ID Health Provider 2:", user_id_health_provider_2)

    # Client test patient stores program
    print("\nClient 1 Storing program...\n")
    action_id = await client_test_patient.store_program(
        cluster_id, CONFIG_PROGRAM_NAME, program_mir_path
    )
    program_id = f"{user_id_test_patient}/{CONFIG_PROGRAM_NAME}"
    print("\nStored program. action_id:", action_id)
    print("\nStored program_id:", program_id)

    # Create secrets for all parties
    #  Patient test party will be a compute time secret
    #  Health provider 1 and 2 will be stored secrets
    print("\nSetting up secrets...\n")
    
    # Initialize secret dictionaries
    party_patient_dict = {}
    party_hp_1_dict = {}
    party_hp_2_dict = {}
    # party_patient_dict[f"patient_image_data_test"] = nillion.SecretInteger(10)
    # party_hp_1_dict[f"hp_1_param_test"] = nillion.SecretInteger(20)
    # party_hp_2_dict[f"hp_2_param_test"] = nillion.SecretInteger(30)

    # Add secret integers to parties
    #  Health provider's dataset sizes
    #  Health provider's theta values
    #  Patient's test image data 
    print("\nhp_1_size and hp_2_size secret integers:")
    print(f"hp_1_size: {len(X_train_subset_sm)}")
    print(f"hp_2_size: {len(X_train_subset_lg)}")
    party_hp_1_dict["hp_1_size"] = nillion.SecretInteger(len(X_train_subset_sm))
    party_hp_2_dict["hp_2_size"] = nillion.SecretInteger(len(X_train_subset_lg))

    print("\nTrain Data Columns:")
    print(train_data.columns.size)
    
    for i in range(train_data.columns.size):
        print(f"x_test{i}: {scaled_test_data[i]}")
        party_patient_dict[f"x_test{i}"] = nillion.SecretInteger(
            scaled_test_data[i]
        )
        print(f"hp1_p{i}: {scaled_theta_subset_sm[i]}")
        party_hp_1_dict[f"hp1_p{i}"] = nillion.SecretInteger(
            scaled_theta_subset_sm[i]
        )
        print(f"hp2_p{i}: {scaled_theta_subset_lg[i]}")
        party_hp_2_dict[f"hp2_p{i}"] = nillion.SecretInteger(
            scaled_theta_subset_lg[i]
        )

    print("\nParty Patient:")
    pprint.PrettyPrinter(indent=4).pprint(party_patient_dict)
    print("\nParty HP 1:")
    pprint.PrettyPrinter(indent=4).pprint(party_hp_1_dict)
    print("\nParty HP 2:")
    pprint.PrettyPrinter(indent=4).pprint(party_hp_2_dict)

    # Health Provider Parties store secrets
    party_hp_1_secrets = nillion.Secrets(party_hp_1_dict)
    party_hp_2_secrets = nillion.Secrets(party_hp_2_dict)

    # Setup default permissions and add compute permissions for test patient on health providers' secrets
    
    # secret_permissions_test_patient = nillion.Permissions.default_for_user(user_id_test_patient)
    
    secret_permissions_hp_1 = nillion.Permissions.default_for_user(user_id_health_provider_1)
    secret_permissions_hp_1.add_compute_permissions({user_id_test_patient: {program_id}})
    
    secret_permissions_hp_2 = nillion.Permissions.default_for_user(user_id_health_provider_2)
    secret_permissions_hp_2.add_compute_permissions({user_id_test_patient: {program_id}})
        
    # Store secrets inputs on the network and retrieve store Ids
    print("\nStoring secrets on the network...\n")
    store_ids = []
    
    print(f"\nStoring secrets for Party HP 1: {party_hp_1_secrets} at program_id: {program_id}")
    program_bindings = nillion.ProgramBindings(program_id)
    program_bindings.add_input_party(
        CONFIG_HEALTH_PROVIDER_PARTIES[0]["party_name"], party_id_health_provider_1
    )
    store_id = await client_health_provider_1.store_secrets(
        cluster_id, program_bindings, party_hp_1_secrets, secret_permissions_hp_1
    )
    store_ids.append(store_id)
    print(f"\nStored Party HP 1 with store_id: {store_id}")
    
    print("\nSleeping for 10 seconds...\n")
    time.sleep(10)
    print("\nWaking up...\n")

    print(f"\nStoring secrets for Party HP 2: {party_hp_2_secrets} at program_id: {program_id}")
    program_bindings = nillion.ProgramBindings(program_id)
    program_bindings.add_input_party(
        CONFIG_HEALTH_PROVIDER_PARTIES[1]["party_name"], party_id_health_provider_2
    )
    store_id = await client_health_provider_2.store_secrets(
        cluster_id, program_bindings, party_hp_2_secrets, secret_permissions_hp_2
    )
    store_ids.append(store_id)
    print(f"\nStored Party HP 2 with store_id: {store_id}")
    
    # Setup compute
    # client_compute = create_nillion_client(
    #     getUserKeyFromFile(CONFIG_TEST_PARTY_1["userkey_file"]),
    #     getNodeKeyFromFile(CONFIG_TEST_PARTY_1["nodekey_alternate_file"]),
    # )
    
    
    # Bind the parties in the computation to the client to set input and output parties
    print("\nSetting up compute bindings..\n")
    
    client_test_compute = create_nillion_client(
        getUserKeyFromFile(CONFIG_TEST_PARTY_1["userkey_file"]),
        getNodeKeyFromFile(CONFIG_TEST_PARTY_1["nodekey_alternate_file"]),
    )
    party_id_test_compute = client_test_compute.party_id()
    user_id_test_compute = client_test_compute.user_id()
    
    print(f"\nComputing on program ID: {program_id} with party ID: {party_id_test_compute}")
    print(f"\nUser ID: {user_id_test_compute}")
    print(f"\nStore IDs: {store_ids}")
    
    compute_bindings = nillion.ProgramBindings(program_id)
    
    compute_bindings.add_input_party(CONFIG_TEST_PARTY_1["party_name"], party_id_test_compute)
    compute_bindings.add_input_party(CONFIG_HEALTH_PROVIDER_PARTIES[0]["party_name"], party_id_health_provider_1)
    compute_bindings.add_input_party(CONFIG_HEALTH_PROVIDER_PARTIES[1]["party_name"], party_id_health_provider_2)
    
    compute_bindings.add_output_party(CONFIG_TEST_PARTY_1["party_name"], party_id_test_compute)
    
    # Setup public variables and compute time secrets
    public_variables = {}
    computation_time_secrets = nillion.Secrets(party_patient_dict)
    
    # Compute the prediction on the secret data
    compute_id = await client_test_compute.compute(
        cluster_id,
        compute_bindings,
        store_ids,
        computation_time_secrets,
        nillion.PublicVariables(public_variables),
    )
    print(f"\nThe computation was sent to the network - compute_id: {compute_id}")
    
    # Compute Results
    print("\nComputing results...\n")
    compute_event_result = await client_test_compute.next_compute_event()
    while True:
        compute_event_result = await client_test_compute.next_compute_event()
        if isinstance(compute_event_result, nillion.ComputeFinishedEvent):
            print(f"✅  Compute complete for compute_id {compute_event_result.uuid}")
            print(f"🖥️  The returned value: {compute_event_result.result.value["patient_test_prediction"]}")
            print(f"🖥️  The returned value: {compute_event_result.result.value["hp_1_data_size"]}")
            print(f"🖥️  The returned value: {compute_event_result.result.value["patient_image_data"]}")
            print(f"Scaling Factor: {scaling_factor}")
            patient_prediction = (
                (compute_event_result.result.value["patient_test_prediction"] / scaling_factor) - 1
            )
            print(f"🔮  The prediction is {patient_prediction}")
            return compute_event_result.result.value


if __name__ == "__main__":
    asyncio.run(main())

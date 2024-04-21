from nada_dsl import *


def nada_main():
    num_params = 31

    # Define the parties
    patient = Party(name="Party1")
    health_provider_1 = Party(name="Party2")
    health_provider_2 = Party(name="Party3")

    # Define the public inputs
    # prediction_init = PublicInteger(Input(name="prediction_init", party=patient))

    # Define the secret inputs
    hp_1_data_size = SecretInteger(Input(name="dataset1_size", party=health_provider_1))
    hp_2_data_size = SecretInteger(Input(name="dataset2_size", party=health_provider_2))
    patient_image_data = []  # Compute time secrets
    hp_1_theta = []  # Stored secrets
    hp_2_theta = []  # Stored secrets

    for i in range(num_params):
        patient_image_data.append(
            SecretInteger(Input(name=f"x_test{i}", party=patient))
        )
        hp_1_theta.append(
            SecretInteger(Input(name=f"hp1_p{i}", party=health_provider_1))
        )
        hp_2_theta.append(
            SecretInteger(Input(name=f"hp2_p{i}", party=health_provider_2))
        )

    ##### TEST ONLY #####
    # patient_test_data = SecretInteger(Input(name="patient_image_data_test", party=patient))
    # party_hp_1_test = SecretInteger(Input(name="hp_1_param_test", party=health_provider_1))
    # party_hp_2_test = SecretInteger(Input(name="hp_2_param_test", party=health_provider_2))
    # prediction = prediction_init
    # prediction = patient_test_data + party_hp_1_test + party_hp_2_test
    ##### END TEST #####

    # Compute weights for each party's data size
    hp_1_weight = hp_1_data_size / (hp_1_data_size + hp_2_data_size)
    hp_2_weight = hp_2_data_size / (hp_1_data_size + hp_2_data_size)

    # # Use weighted average to compute the combined theta
    combined_theta = []
    for hp_1_param, hp_2_param in zip(hp_1_theta, hp_2_theta):
        combined_theta.append((hp_1_param * hp_1_weight) + (hp_2_param * hp_2_weight))

    # Perform element-wise multiplication and summation for scaled data
    prediction = Integer(0)
    for x_val, theta_val in zip(patient_image_data, combined_theta):
        prediction += x_val * theta_val

    # prediction = hp_1_data_size

    return [Output(prediction, "patient_test_prediction", patient)]

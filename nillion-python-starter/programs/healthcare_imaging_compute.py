from nada_dsl import *


def nada_main():
    num_coefficients = 31

    # Define the parties
    patient = Party(name="Patient_test_img")
    health_provider_1 = Party(name="HP_1")
    health_provider_2 = Party(name="HP_2")

    # Define the secret inputs
    hp_1_data_size = SecretInteger(Input(name="hp_1_data_size", party=health_provider_1))
    hp_2_data_size = SecretInteger(Input(name="hp_2_data_size", party=health_provider_2))
    patient_image_data = [] # Compute time secrets
    hp_1_thetas = []  # Stored secrets
    hp_2_thetas = []  # Stored secrets  

    for i in range(1, num_coefficients):
        patient_image_data.append(
            SecretInteger(Input(name="patient_image_data_"+str(i), party=patient))
        )
        hp_1_thetas.append(
            SecretInteger(Input(name="hp_1_coeff_"+str(i), party=health_provider_1))
        )
        hp_1_thetas.append(
            SecretInteger(Input(name="hp_2_coeff_"+str(i), party=health_provider_2))
        )

    # Initialize the prediction
    prediction = 0
    
    # Compute weights for each party's data size
    hp_1_weight = hp_1_data_size / (hp_1_data_size + hp_2_data_size)
    hp_2_weight = hp_2_data_size / (hp_1_data_size + hp_2_data_size)
    
    # Use weighted average to compute thetas
    combined_theta = (hp_1_thetas * hp_1_weight) + (hp_2_thetas * hp_2_weight)
    
    # Perform element-wise multiplication and summation for scaled data
    for x_val, theta_val in zip(patient_image_data, combined_theta):
        prediction += x_val * theta_val
    
    return [Output(prediction, "patient_prediction", patient)]

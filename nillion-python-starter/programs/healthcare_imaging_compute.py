from nada_dsl import *


def nada_main():
    num_params = 31

    # Define the parties
    patient = Party(name="Patient_test_img")
    health_provider_1 = Party(name="HP_1")
    health_provider_2 = Party(name="HP_2")
    
    # Define the public inputs
    prediction_init = PublicInteger(Input(name="prediction_init", party=patient))

    # Define the secret inputs
    hp_1_data_size = SecretInteger(Input(name="hp_1_data_size", party=health_provider_1))
    hp_2_data_size = SecretInteger(Input(name="hp_2_data_size", party=health_provider_2))
    patient_image_data = [] # Compute time secrets
    hp_1_theta = []  # Stored secrets
    hp_2_theta = []  # Stored secrets  

    for i in range(1, num_params):
        patient_image_data.append(
            SecretInteger(Input(name="patient_image_data_"+str(i), party=patient))
        )
        hp_1_theta.append(
            SecretInteger(Input(name="hp_1_param_"+str(i), party=health_provider_1))
        )
        hp_1_theta.append(
            SecretInteger(Input(name="hp_2_param_"+str(i), party=health_provider_2))
        )

        
    # Compute weights for each party's data size
    hp_1_weight = hp_1_data_size / (hp_1_data_size + hp_2_data_size)
    hp_2_weight = hp_2_data_size / (hp_1_data_size + hp_2_data_size)
    
    # Use weighted average to compute the combined theta
    combined_theta = []
    for hp_1_param, hp_2_param in zip(hp_1_theta, hp_2_theta):
        combined_theta.append((hp_1_param * hp_1_weight) + (hp_2_param * hp_2_weight)) 
    
    # Perform element-wise multiplication and summation for scaled data
    prediction = prediction_init
    for x_val, theta_val in zip(patient_image_data, combined_theta):
        prediction += x_val * theta_val
    
    return [Output(prediction, "patient_test_prediction", patient)]

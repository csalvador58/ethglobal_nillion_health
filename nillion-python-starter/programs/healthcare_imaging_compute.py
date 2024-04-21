from nada_dsl import *


def nada_main():
    num_params = 31

    # Define the parties
    patient = Party(name="Patient_test_img")
    health_provider_1 = Party(name="HP_1")
    health_provider_2 = Party(name="HP_2")

    # Define the public inputs
    # prediction_init = PublicInteger(Input(name="prediction_init", party=patient))

    # Define the secret inputs
    hp_1_data_size = SecretInteger(
        Input(name="hp_1_size", party=health_provider_1)
    )
    hp_2_data_size = SecretInteger(
        Input(name="hp_2_size", party=health_provider_2)
    )
    patient_image_data = []  # Compute time secrets
    hp_1_theta = []  # Stored secrets
    hp_2_theta = []  # Stored secrets

    for i in range(num_params):
        patient_image_data.append(
            SecretInteger(Input(name=f"x_test{i}", party=patient))
        )
    for i in range(num_params):
        hp_1_theta.append(
            SecretInteger(Input(name=f"hp1_p{i}", party=health_provider_1))
        )
    for i in range(num_params):
        hp_2_theta.append(
            SecretInteger(Input(name=f"hp2_p{i}", party=health_provider_2))
        )
    # # Manual add for patient_image_data from 0 to 30
    # patient_image_data.append(SecretInteger(Input(name="x_test0", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test1", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test2", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test3", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test4", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test5", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test6", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test7", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test8", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test9", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test10", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test11", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test12", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test13", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test14", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test15", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test16", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test17", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test18", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test19", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test20", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test21", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test22", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test23", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test24", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test25", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test26", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test27", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test28", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test29", party=patient)))
    # patient_image_data.append(SecretInteger(Input(name="x_test30", party=patient)))

    # # Manual add for hp_1_theta from 0 to 30
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p0", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p1", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p2", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p3", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p4", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p5", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p6", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p7", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p8", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p9", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p10", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p11", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p12", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p13", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p14", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p15", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p16", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p17", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p18", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p19", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p20", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p21", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p22", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p23", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p24", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p25", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p26", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p27", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p28", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p29", party=health_provider_1)))
    # hp_1_theta.append(SecretInteger(Input(name="hp1_p30", party=health_provider_1)))

    # # Manual add for hp_2_theta from 0 to 30
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p0", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p1", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p2", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p3", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p4", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p5", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p6", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p7", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p8", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p9", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p10", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p11", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p12", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p13", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p14", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p15", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p16", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p17", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p18", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p19", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p20", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p21", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p22", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p23", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p24", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p25", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p26", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p27", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p28", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p29", party=health_provider_2)))
    # hp_2_theta.append(SecretInteger(Input(name="hp2_p30", party=health_provider_2)))

    ##### TEST ONLY #####
    # patient_test_data = SecretInteger(Input(name="patient_image_data_test", party=patient))
    # party_hp_1_test = SecretInteger(Input(name="hp_1_param_test", party=health_provider_1))
    # party_hp_2_test = SecretInteger(Input(name="hp_2_param_test", party=health_provider_2))
    # prediction = prediction_init
    # prediction = patient_test_data + party_hp_1_test + party_hp_2_test
    ##### END TEST #####

    # # Compute weights for each party's data size
    # hp_1_weight = hp_1_data_size / (hp_1_data_size + hp_2_data_size)
    # hp_2_weight = hp_2_data_size / (hp_1_data_size + hp_2_data_size)

    # # # Use weighted average to compute the combined theta
    # combined_theta = []
    # for hp_1_param, hp_2_param in zip(hp_1_theta, hp_2_theta):
    #     combined_theta.append((hp_1_param * hp_1_weight) + (hp_2_param * hp_2_weight)) 
    
    # # Perform element-wise multiplication and summation for scaled data
    # prediction = Integer(0)
    # for x_val, theta_val in zip(patient_image_data, combined_theta):
    #     prediction += x_val * theta_val
        
    prediction = hp_1_data_size
    
    return [
        Output(prediction, "patient_test_prediction", patient),
        Output(hp_1_data_size, "hp_1_data_size", patient),
        Output(patient_image_data[0], "patient_image_data", patient),
        ]

from api.app import Patient

def test_patient_schema_valid():
    # should not raise
    Patient(time_in_hospital=3, num_medications=10, number_diagnoses=5)

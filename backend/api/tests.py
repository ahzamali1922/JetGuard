from unittest.mock import patch

from rest_framework.test import APITestCase


class FakePredictor:
    def predict(self, observations):
        return 42.5

    def public_info(self):
        return {"model": "Tuned LSTM", "window_length": 50}


def make_observation(cycle):
    observation = {
        "cycle": cycle,
        "setting_1": 0.1,
        "setting_2": 0.2,
        "setting_3": 0.3,
    }
    observation.update({f"sensor_{index}": float(index) for index in range(1, 22)})
    return observation


class ApiEndpointTests(APITestCase):
    @patch("api.views.get_predictor", return_value=FakePredictor())
    def test_health_reports_model_ready(self, _get_predictor):
        response = self.client.get("/api/health/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_loaded": True})

    @patch("api.views.get_predictor", return_value=FakePredictor())
    def test_model_info_returns_selected_model(self, _get_predictor):
        response = self.client.get("/api/model-info/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["window_length"], 50)

    @patch("api.views.get_predictor", return_value=FakePredictor())
    def test_prediction_accepts_valid_engine_observations(self, _get_predictor):
        response = self.client.post(
            "/api/predict/",
            {
                "engine_id": 7,
                "observations": [make_observation(1), make_observation(2)],
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["engine_id"], 7)
        self.assertEqual(response.json()["final_cycle"], 2)
        self.assertEqual(response.json()["predicted_rul_cycles"], 42.5)

    @patch("api.views.get_predictor", return_value=FakePredictor())
    def test_prediction_rejects_non_increasing_cycles(self, _get_predictor):
        response = self.client.post(
            "/api/predict/",
            {
                "observations": [make_observation(2), make_observation(2)],
            },
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("observations", response.json())

    @patch("api.views.get_predictor", return_value=FakePredictor())
    def test_prediction_rejects_missing_sensor_values(self, _get_predictor):
        observation = make_observation(1)
        del observation["sensor_21"]
        response = self.client.post(
            "/api/predict/",
            {"observations": [observation]},
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("sensor_21", str(response.json()["observations"]))

import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from api.inference import ModelArtifactError, get_predictor
from api.serializers import PredictRequestSerializer


logger = logging.getLogger(__name__)


class HealthView(APIView):
    def get(self, request):
        try:
            get_predictor()
        except ModelArtifactError:
            logger.exception("JetGuard model is unavailable.")
            return Response(
                {"status": "unavailable", "model_loaded": False},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response({"status": "ok", "model_loaded": True})


class ModelInfoView(APIView):
    def get(self, request):
        try:
            info = get_predictor().public_info()
        except ModelArtifactError:
            logger.exception("JetGuard model information is unavailable.")
            return Response(
                {"error": "The trained model is unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(info)


class PredictView(APIView):
    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data

        try:
            predictor = get_predictor()
            prediction = predictor.predict(payload["observations"])
        except ModelArtifactError:
            logger.exception("JetGuard prediction failed because model artifacts are unavailable.")
            return Response(
                {"error": "The trained model is unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        response = {
            "predicted_rul_cycles": prediction,
            "final_cycle": payload["observations"][-1]["cycle"],
            "model": "Tuned LSTM",
            "window_length": 50,
        }
        if "engine_id" in payload:
            response["engine_id"] = payload["engine_id"]
        return Response(response)

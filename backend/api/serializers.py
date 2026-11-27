from rest_framework import serializers


class ObservationSerializer(serializers.Serializer):
    cycle = serializers.IntegerField(min_value=1)
    setting_1 = serializers.FloatField()
    setting_2 = serializers.FloatField()
    setting_3 = serializers.FloatField()
    sensor_1 = serializers.FloatField()
    sensor_2 = serializers.FloatField()
    sensor_3 = serializers.FloatField()
    sensor_4 = serializers.FloatField()
    sensor_5 = serializers.FloatField()
    sensor_6 = serializers.FloatField()
    sensor_7 = serializers.FloatField()
    sensor_8 = serializers.FloatField()
    sensor_9 = serializers.FloatField()
    sensor_10 = serializers.FloatField()
    sensor_11 = serializers.FloatField()
    sensor_12 = serializers.FloatField()
    sensor_13 = serializers.FloatField()
    sensor_14 = serializers.FloatField()
    sensor_15 = serializers.FloatField()
    sensor_16 = serializers.FloatField()
    sensor_17 = serializers.FloatField()
    sensor_18 = serializers.FloatField()
    sensor_19 = serializers.FloatField()
    sensor_20 = serializers.FloatField()
    sensor_21 = serializers.FloatField()


class PredictRequestSerializer(serializers.Serializer):
    engine_id = serializers.IntegerField(min_value=1, required=False)
    observations = ObservationSerializer(many=True, allow_empty=False, max_length=5000)

    def validate_observations(self, observations):
        cycles = [observation["cycle"] for observation in observations]
        if any(current <= previous for previous, current in zip(cycles, cycles[1:])):
            raise serializers.ValidationError("Cycles must be strictly increasing.")
        return observations

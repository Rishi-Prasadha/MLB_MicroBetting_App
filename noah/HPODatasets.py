""" Airline Dataset target label and feature column names  """
airline_label_column = "ArrDel15"
airline_feature_columns = [
    "Year",
    "Quarter",
    "Month",
    "DayOfWeek",
    "Flight_Number_Reporting_Airline",
    "DOT_ID_Reporting_Airline",
    "OriginCityMarketID",
    "DestCityMarketID",
    "DepTime",
    "DepDelay",
    "DepDel15",
    "ArrDel15",
    "AirTime",
    "Distance",
]
airline_dtype = "float32"

""" NYC TLC Trip Record Data target label and feature column names  """
nyctaxi_label_column = "above_average_tip"
nyctaxi_feature_columns = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "above_average_tip",
]
nyctaxi_dtype = "float32"


""" Insert your dataset here! """

BYOD_label_column = "pitch_type"  # e.g., nyctaxi_label_column
BYOD_feature_columns = [
 "stand_L",
 "previous_zone_A",
 "previous_zone_B",
 "previous_zone_C",
 "previous_zone_D",
 "previous_zone_E",
 "previous_zone_F",
 "previous_zone_G",
 "previous_zone_H",
 "previous_zone_I",
 "previous_zone_K",
 "previous_zone_L",
 "previous_zone_M",
 "previous_zone_N",
 "previous_pitch_type_CH",
 "previous_pitch_type_CU",
 "previous_pitch_type_FF",
 "previous_pitch_type_SL",
 "previous_description_ball",
 "previous_description_blocked_ball",
 "previous_description_called_strike",
 "previous_description_foul",
 "previous_description_hit_into_play",
 "previous_description_swinging_strike",
 "previous_event_field_out",
 "previous_event_single",
 "previous_event_strikeout",
 "previous_event_walk",
 "previous_bb_type_fly_ball",
 "previous_bb_type_ground_ball",
 "previous_bb_type_line_drive",
 "previous_bb_type_popup",
 "outs_when_up",
 "1st_pitch",
 "pitch_count",
 "score_diff",
 "count",
 "bases",
 "pitch_type",
 ]  # e.g., nyctaxi_feature_columns

BYOD_dtype = "float32"  # e.g., nyctaxi_dtype


SERVER_HOST = "localhost"

#Settings for MongoDB
MONGO_SERVER_HOST = SERVER_HOST
MONGO_SERVER_PORT = 27017
#MONGO_DB = "LIVIS"
MONGO_DB = "parts"
INSPECTION_DATA_COLLECTION = "inspection_summary"
MONGO_COLLECTION_PARTS = "parts"
MONGO_COLLECTIONS = {MONGO_COLLECTION_PARTS: "parts"}
WORKSTATION_COLLECTION = 'workstations'
PARTS_COLLECTION = 'parts'
SHIFT_COLLECTION = 'shift'
# PLAN_COLLECTION = 'plan'


# #Settings for Redis
REDIS_CLIENT_HOST = "localhost"
REDIS_CLIENT_PORT = 6379

# MONGO_DB = "Indo_trial"
# INSPECTION_DATA_COLLECTION = "inspection_summary"
# MONGO_COLLECTION_PARTS = "parts"
# MONGO_COLLECTIONS = {"MONGO_COLLECTION_PARTS": "parts" ,"INSPECTION_DATA_COLLECTION" :"inspection_summary" ,"SHIFT_COLLECTION" : 'shift'}
# WORKSTATION_COLLECTION = 'workstation'
# PARTS_COLLECTION = 'parts'
# SHIFT_COLLECTION = 'shift'

original_frame_keyholder = "original_frame"
predicted_frame_keyholder = "predicted_frame"

#mobile_id = '865cd46bb65045eb' # krishna
#mobile_id = 'f2d3eb4dab865972' # manju
# mobile_id = '3e9421a611ed7e30' # richa
#mobile_id = 'c9ba534e0e016266' # vinayaka


#mobile_id = '16c61100188098ee'#non se mobile on non se n/w -> router ax router wifi used for data capture 
mobile_id = 'f3e76c13fd071188'#semobile on se n/w  

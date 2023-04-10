import json
import pickle
import os, sys

towns = ["01", "02", "03", "04","05", "06", "07", "10"]
index_per_down = ["val"] + [str(_).zfill(2) for _ in range(0, 11)]
data_folder = "../"

json_path = "../collect_data_json"


## Store the length of each route since we truncate those ones with infraction happening 
length_dict = {}
for town_name in towns:
    for data_index in index_per_down:
        length_dict["town"+town_name+"_"+data_index] = {}
        try:
            with open(os.path.join(json_path,  "data_collect_town"+town_name+"_"+data_index+"_results.json", ), "r") as f:
                records = json.load(f)["_checkpoint"]["records"]
        except:
            print("No json for", os.path.join(json_path,  "data_collect_town"+town_name+"_"+data_index+"_results.json", ), flush=True)
            continue
        for index, record in enumerate(records):
            record = records[index]
            route_data_folder = os.path.join(data_folder, "town"+town_name+"_"+data_index, record["meta"]["folder_name"])

            measurements_files = os.listdir(os.path.join(route_data_folder, "measurements"))
            total_length = len(measurements_files)
            if record["scores"]["score_composed"] >= 100:
                length_dict["town"+town_name+"_"+data_index][route_data_folder] = total_length
                continue

            # timeout or blocked, remove the last ones where the vehicle stops
            if len(record["infractions"]["route_timeout"]) > 0 or \
                len(record["infractions"]["vehicle_blocked"]) > 0:
                    stop_index = 0
                    for i in range(total_length-1, 0, -1):
                        fname = os.path.join(route_data_folder, "measurements", str(i).zfill(4)) + ".json"
                        with open(fname, 'r') as mf:
                            speed = json.load(mf)["speed"]
                        if speed > 0.1:
                            stop_index = i
                            break
                    length_dict["town"+town_name+"_"+data_index][route_data_folder] = min(total_length, stop_index + 5)
            # collision or red-light
            elif len(record["infractions"]["red_light"]) > 0 or \
                len(record["infractions"]["collisions_pedestrian"]) > 0 or \
                len(record["infractions"]["collisions_vehicle"]) > 0 or \
                len(record["infractions"]["collisions_layout"]) > 0:
                length_dict["town"+town_name+"_"+data_index][route_data_folder] = max(0, total_length-10)
        print(os.path.join(json_path,  "data_collect_town"+town_name+"_"+data_index+"_results.json", ), "Routes with Jsons Finished", flush=True)


#### Data without any json file
all_town_folder =  os.listdir(data_folder)
for town_name in towns:
    for data_index in index_per_down:
        if "town"+town_name+"_"+data_index in all_town_folder:
            all_route_folder = os.listdir(os.path.join(data_folder, "town"+town_name+"_"+data_index))
            for route_folder in all_route_folder:
                route_data_folder = os.path.join(data_folder, "town"+town_name+"_"+data_index, route_folder)
                if route_data_folder in length_dict["town"+town_name+"_"+data_index]:
                    continue

                measurements_files = os.listdir(os.path.join(route_data_folder, "measurements"))
                total_length = len(measurements_files)
                if total_length > 10:
                    total_length = total_length - 10
                    stop_index = 0
                    for i in range(total_length-1, 0, -1):
                        fname = os.path.join(route_data_folder, "measurements", str(i).zfill(4)) + ".json"
                        with open(fname, 'r') as mf:
                            speed = json.load(mf)["speed"]
                        if speed > 0.1:
                            stop_index = i
                            break
                    length_dict["town"+town_name+"_"+data_index][route_data_folder] = min(total_length, stop_index + 5)
            print(os.path.join(json_path,  "data_collect_town"+town_name+"_"+data_index+"_results.json", ), "Routes without Jsons Finished", flush=True)

print("Town-Name    #Route-Per-Town    #Frames   -------------------")
total_cnt = 0
for town_name in length_dict:
    cnt = 0
    for number_of_frames_per_route in length_dict[town_name].values():
        cnt += number_of_frames_per_route
    print(town_name,  length_dict[town_name], cnt)
    total_cnt += cnt
print("Total Frames:", total_cnt)


with open("../dataset_metadata.pkl", "wb") as f:
    pickle.dump(length_dict, f)

print("Dataset metadata pickle saved")



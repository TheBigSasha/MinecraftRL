import csv
import numpy as np

def load_item_mapping_from_csv(file_path):
    item_mapping = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row if it exists
        for row in reader:
            item_name, _, item_id, item_id_2 = row  # Change this line to read the first and third columns
            if item_id is None or item_id == '':
                item_id = 10000 + len(item_mapping)
            item_id = int(item_id)
            if item_id_2 is not None:
                item_id = item_id + 10000 * int(item_id_2)

            item_mapping[item_name] = item_id
            item_mapping[item_name.replace(" ", "_")] = item_id
            item_mapping[item_name.replace(" ", "_").lower()] = item_id
            item_mapping[item_name.replace(" ", "_").upper()] = item_id
            item_mapping[item_name.replace(" ", "_").capitalize()] = item_id
            item_mapping[item_name.replace(" ", "_").lower().capitalize()] = item_id
            item_mapping[item_name.replace(" ", "_").upper().capitalize()] = item_id
            item_mapping[item_id] = item_id
    return item_mapping

def preprocess_observation(obs_dict, string_to_index_mapping):
    rgb = obs_dict['rgb']
    # equipment = obs_dict['equipment']["name"]
    # equipment_count = obs_dict['equipment']["quantity"]
    # inventory = obs_dict['inventory']['name']
    # inventory_count = obs_dict['inventory']['quantity']
    # block_names = obs_dict["voxels"]["block_name"]
    # block_collidable = obs_dict["voxels"]["is_collidable"]
    # tool_req = obs_dict["voxels"]["is_tool_not_required"]
    # is_liquid = obs_dict["voxels"]["is_liquid"]
    # is_solid = obs_dict["voxels"]["is_solid"]
    # look_vector = obs_dict["voxels"]["cos_look_vec_angle"]


    # give us a 1d array of all this stuff smooshed into ints
    obs_list = []

    # rgb_flattened = np.zeros((len(rgb[0]), len(rgb[0][0])), dtype=np.uint32)
    # for channel in rgb:
    #     for idx, row in enumerate(channel):
    #         for idx2, item in enumerate(row):
    #             rgb_flattened[idx][idx2] = rgb_flattened[idx][idx2] << 8
    #             rgb_flattened[idx][idx2] = rgb_flattened[idx][idx2] | item

    # for row in rgb_flattened:
    #     for item in row:
    #         obs_list.append(item)
    # for item in equipment:
    #     obs_list.append(string_to_index_mapping[item])
    # for item in equipment_count:
    #     obs_list.append(item)
    # for item in inventory:
    #     obs_list.append(string_to_index_mapping[item])
    # for item in inventory_count:
    #     obs_list.append(item)
    # for list in block_names:
    #     for sub_list in list:
    #         for item in sub_list:
    #             obs_list.append(string_to_index_mapping[item])
    # block_collidable_int = 0
    # for item in block_collidable:
    #     # 2 nested lists inside this, but we want to flatten them each to ints using bit shifting
    #     for sub_list in item:
    #         for sub_item in sub_list:
    #             block_collidable_int = block_collidable_int << 1
    #             block_collidable_int = block_collidable_int | sub_item
    # obs_list.append(block_collidable_int)
    # tool_req_int = 0
    # for item in tool_req:
    #     # 2 nested lists inside this, but we want to flatten them each to ints using bit shifting
    #     for sub_list in item:
    #         for sub_item in sub_list:
    #             tool_req_int = tool_req_int << 1
    #             tool_req_int = tool_req_int | sub_item
    # obs_list.append(tool_req_int)
    # is_liquid_int = 0
    # for item in is_liquid:
    #     # 2 nested lists inside this, but we want to flatten them each to ints using bit shifting
    #     for sub_list in item:
    #         for sub_item in sub_list:
    #             is_liquid_int = is_liquid_int << 1
    #             is_liquid_int = is_liquid_int | sub_item
    # obs_list.append(is_liquid_int)
    # is_solid_int = 0
    # for item in is_solid:
    #     # 2 nested lists inside this, but we want to flatten them each to ints using bit shifting
    #     for sub_list in item:
    #         for sub_item in sub_list:
    #             is_solid_int = is_solid_int << 1
    #             is_solid_int = is_solid_int | sub_item
    # obs_list.append(is_solid_int)
    # for item in look_vector:
    #     for sub_item in item:
    #         for sub_sub_item in sub_item:
    #             obs_list.append(sub_sub_item)
    #
    # # Data type: numpy.float32
    # # Shape: (6,)
    # for item in cur_durability:
    #     obs_list.append(item)
    #
    # for item in max_durability:
    #     obs_list.append(item)
    #
    # for item in inc_name_by_craft:
    #     obs_list.append(string_to_index_mapping[item])
    # for item in inc_quantity_by_craft:
    #     obs_list.append(item)
    #
    # for item in inc_name_by_other:
    #     obs_list.append(string_to_index_mapping[item])
    # for item in inc_quantity_by_other:
    #     obs_list.append(item)
    #
    # for item in dec_name_by_craft:
    #     obs_list.append(string_to_index_mapping[item])
    # for item in dec_quantity_by_craft:
    #     obs_list.append(item)
    #
    # for item in dec_name_by_other:
    #     obs_list.append(string_to_index_mapping[item])
    # for item in dec_quantity_by_other:
    #     obs_list.append(item)
    #
    # for item in life:
    #     obs_list.append(item)
    #
    # for item in oxygen:
    #     obs_list.append(item)
    #
    # for item in armor:
    #     obs_list.append(item)
    #
    # for item in food:
    #     obs_list.append(item)
    #
    # for item in saturation:
    #     obs_list.append(item)
    #
    # obs_list.append(is_sleeping)
    #
    # for item in xp:
    #     obs_list.append(item)
    #
    #
    # for item in pos:
    #     obs_list.append(item)

    return rgb


def get_args_mask_for_func_action(fun_act_idx) -> str:
    if (fun_act_idx == 0):
        return 'no-op'
    elif (fun_act_idx == 1):
        return 'use'
    elif (fun_act_idx == 2):
        return 'drop'
    elif (fun_act_idx == 3):
        return 'attack'
    elif (fun_act_idx == 4):
        return 'craft_smelt'
    elif (fun_act_idx == 5):
        return 'equip'
    elif (fun_act_idx == 6):
        return 'place'
    elif (fun_act_idx == 7):
        return 'destroy'
    else:
        return 'no-op'

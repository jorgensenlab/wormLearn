import torch
import time
import numpy as np
import skimage.filters
from scipy import ndimage
import math
import os
from PIL import Image
from PIL.ImageOps import invert, autocontrast
import glob
from cv2 import cv2
import datetime
import re
import pandas as pd
from . import IO_tools
from . import vis_tools
from . import utils


class InferenceManager():
    '''
    Container class. Hand it input data, it wil organize it into experiments. 
    Inferences, processing, and requests for data are properly handed to each experiment
    '''
    def __init__(self, path_to_data, path_to_model, file_type, ts_fmt=None):
        self.path_to_data = path_to_data
        self.path_to_model = path_to_model
        self.file_type = file_type   ## this should include the dot: ".tif"
        self.ts_fmt = ts_fmt
        self.groups = None
        
    def read_data(self, stack_preprocess_data=False, subfolders_too=False, folders_are_groups_subfolders_are_expmts=False):
        if subfolders_too:
            print('InferenceManager.read_data subfolders_too has been removed!')
        data_folders = glob.glob(self.path_to_data + "/*/")
        if stack_preprocess_data:
            datasets = [IO_tools.ImageStackset(path, self.path_to_data, IO_tools.inference_transforms(), self.ts_fmt, file_type=self.file_type) for path in data_folders]
        else:
            if folders_are_groups_subfolders_are_expmts:
                datasets = [IO_tools.Imageset(subfolder, self.path_to_data,  IO_tools.inference_transforms(), img_type=self.file_type) for path in data_folders for subfolder in glob.glob(path + "/*/")]
                self.groups = [['\\' +os.path.basename(os.path.normpath(path)) + '\\' + os.path.basename(os.path.normpath(subfolder)) + '\\' for subfolder in glob.glob(path + "/*/")] for path in data_folders ]
                self.group_names = [os.path.basename(os.path.normpath(path)) for path in data_folders]
                print(str(len(self.groups)) + ' Groups:')
                [print("Group", n, "=", g) for n, g in zip(self.group_names, self.groups)]
            else:
                datasets = [IO_tools.Imageset(path, self.path_to_data,  IO_tools.inference_transforms(), img_type=self.file_type) for path in data_folders]
        self.experiments = [Experiment(dataset, self.file_type) for dataset in datasets]
    
    def infer_worm_BBoxes(self):
        self._infer_worm_BBoxes()
        self.organize_inference_results()
    
    def _infer_worm_BBoxes(self):
        ##### Set up model
        print('Loading model...')
        num_classes = 2
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        cpu_device = torch.device("cpu")
        
        model = utils.get_instance_BB_model(num_classes)
        model.to(device)
        model.load_state_dict(torch.load(self.path_to_model, map_location=device))
        model.eval()
        torch.no_grad()
        torch.set_num_threads(1)
        
        for experiment in self.experiments:
            dataset = experiment.imageset
            print('Infering on dataset ' + dataset.root + ' ...')
            batch_size = 1 
            # dataset = utils.ImageDataset(path, utils.inference_transforms(invert=need_invert))
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            reader = IO_tools.DatasetReader(data_loader)
            
            #### Infer on images
            i=1
            inference_results = {}
            all_out = []
            for images, paths in reader.serve_images():
                images = list(image.to(device) for image in images)  ## Move images to GPU
                
                outputs = model(images)  ## This is the inference 
                
                outputs_cpu = [{k: v.to(cpu_device).detach().numpy() for k, v in t.items()} for t in outputs] ## Move results back to CPU
                for i in range(len(outputs_cpu)):
                    inference_results[paths[i]] = outputs_cpu[i]
            experiment.store_inference_results(inference_results)
    
    def organize_inference_results(self):
        [experiment.organize_inference_results(self.ts_fmt) for experiment in self.experiments]
    
    def track_worms(self, pixels_per_mm=None, framerate=2, missing_frames_forgiven=0, max_mm_per_s=2, frame_window=None):
        if self.check_for_calibrations():
            self.localize_worms()
            self.track_worms_within_experiment_timepoints(pixels_per_mm=pixels_per_mm, 
                                                            framerate=framerate, 
                                                            missing_frames_forgiven=missing_frames_forgiven, 
                                                            max_mm_per_s=max_mm_per_s, 
                                                            frame_window=frame_window)
    
    def check_for_calibrations(self):
        ok_go = True
        for expmt in self.experiments:
            if expmt.pixels_per_mm is None:
                print(expmt.imageset.root_stub + ' experiment missing calibration. Please fix this.')
                ok_go = False
        return ok_go
    
    
    def localize_worms(self):
        for experiment in self.experiments:
            for timepoint in experiment:
                for img_path, result in timepoint:
                    img = experiment.imageset.get_by_name_as_gray_Image(img_path)
                    CoM = []
                    for i in range(result['boxes'].shape[0]):
                        left   = int(result['boxes'][i, 0])
                        top    = int(result['boxes'][i, 1])
                        right  = int(result['boxes'][i, 2])
                        bottom = int(result['boxes'][i, 3])
                        score  = result['scores'][i]
                        
                        cropped = img.crop((left, top, right, bottom))
                        center_of_mass, _ = get_center_of_worm(cropped)
                        center_of_mass_global = (center_of_mass[1] + left , center_of_mass[0] + top)
                        CoM.append(center_of_mass_global)
                    result['CoM'] = np.array(CoM, dtype='float32')
    
    def track_worms_within_experiment_timepoints(self, pixels_per_mm=None, framerate=2, missing_frames_forgiven=0, max_mm_per_s=2, frame_window=None):
        [experiment.track_worms_within_timepoints(pixels_per_mm=pixels_per_mm, 
                                                  framerate=framerate, 
                                                  missing_frames_forgiven=missing_frames_forgiven, 
                                                  max_mm_per_s=max_mm_per_s,
                                                  frame_window=frame_window) 
                                                    for experiment in self.experiments]
    
    def save_as_PASCAL_VOC(self):
        for experiment in self.experiments:
            experiment.save_as_PASCAL_VOC()

    
    def load_inferences_from_xml(self):
        self._load_inferences_from_PASCAL_VOC()
        self.organize_inference_results()
    
    def _load_inferences_from_PASCAL_VOC(self):
        for experiment in self.experiments:
            loaded_results = {}
            for img_path in experiment.imageset.imgs:
                annotationFilePath = os.path.splitext(img_path)[0] + '.xml'
                loaded_results[img_path] = IO_tools.PascalVOCReader(annotationFilePath).getInference()
            experiment.store_inference_results(loaded_results)
    
    def stretch_boxes(self, stretch):
        for experiment in self.experiments:
            for timepoint in experiment:
                for img_path, results in timepoint:
                    results['boxes'] = results['boxes'] + np.array([[-stretch, -stretch, stretch, stretch], ] * results['boxes'].shape[0])
    
    # def attach_calibrations(self, px=None, mm=None, rig_plate=None, rig_pxmm=None):
    #     if rig_plate is not None:
    #         for rig in rig_plate:
    #     else
    #         [experiment.set_calibration(p / m) for experiment, p, m, in zip(self.experiments, px, mm)]
            
    def calibration_dictionary(self, calibrations):
        '''
        calibrations: a dict of {px/mm: [plate1, plate2, ...]}
        '''
        for px_per_mm, plates in calibrations.items():
            for plate in plates:
                try:
                    self.get_Experiment_by_data_folder(plate).set_calibration(px_per_mm)
                    print('set  ' + plate + ' as ' + str(px_per_mm))
                except AttributeError:
                    print('No calibration set for ' + plate + '. No such experiment found.')
    
    def all_calibrations_the_same(self, px_per_mm):
        for expmt in self.experiments:
            expmt.set_calibration(px_per_mm)
        print('Set calibrations')

    def mean_speeds_by_group(self, groups=None, group_names=None, min_frames=2, framerate=2):
        def df_cleanup(df):
            df = df.astype('float64')
            df['time'] = df.index
            df['time'] = df['time'].apply(lambda x:  pd.Timestamp(datetime.datetime.strptime(x, '%d:%H:%M:%S')), 1)
            return df
        
        speeds_by_group = {}
        SEMs_by_group = {}
        N_frames_per_capture = 0
        all_timepoints = set()
        if groups is None:
            groups = self.groups
        if group_names is None:
            group_names = self.group_names
        for group in groups:
            for member in group:
                experiment = self.get_Experiment_by_data_folder(member)
                if experiment is not None:
                    for timepoint in experiment:
                        N_frames_per_capture = max(N_frames_per_capture, timepoint.frame_window_length)
                    all_timepoints |= set(experiment.timepoints.keys())
        all_timepoints = sorted(list(all_timepoints))
        means_df = pd.DataFrame(index=all_timepoints, columns=group_names)
        SEMs_df = means_df.copy()
        N_df = means_df.copy()
        for group, group_name in zip(groups, group_names):
            for timepoint in all_timepoints:
                speeds = []
                frame_deltas = []
                for member in group:
                    member_experiment = self.get_Experiment_by_data_folder(member)
                    if member_experiment is not None:
                        try:
                            speed, frame_delta = member_experiment.timepoints[timepoint].all_speeds(member_experiment.pixels_per_mm, min_frames=min_frames, framerate=framerate)
                        except KeyError:
                            pass
                        speeds = speeds + speed
                        frame_deltas = frame_deltas + frame_delta
                N_estimator = sum(frame_deltas) / N_frames_per_capture
                weighted_mean_speed = sum(x * y for x, y in zip(speeds, frame_deltas)) / sum(frame_deltas)
                weighted_variance = sum( f * (s - weighted_mean_speed) ** 2 for s, f in zip(speeds, frame_deltas) ) / sum(frame_deltas) # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
                weighted_stdDev = math.sqrt(weighted_variance)
                weighted_SEM = weighted_stdDev / math.sqrt(sum(frame_deltas) / N_frames_per_capture)
                means_df.loc[timepoint, group_name] = weighted_mean_speed
                SEMs_df.loc[timepoint, group_name] = weighted_SEM
                N_df.loc[timepoint, group_name] = N_estimator
        means_df = df_cleanup(means_df)
        SEMs_df = df_cleanup(SEMs_df)
        N_df = df_cleanup(N_df)
        return means_df, SEMs_df, N_df

    def speeds_by_group(self, groups, group_names, min_frames=2, framerate=2):
        speeds_by_group = {}
        for n, group in enumerate(groups):
            speeds_by_group[group_names[n]] = []
            N_timepoints = max([len(self.get_Experiment_by_data_folder(folder)) for folder in group])
            for i in range(N_timepoints):
                speeds = []
                frame_deltas = []
                for member in group:
                    member_experiment = self.get_Experiment_by_data_folder(member)
                    try:
                        speed, frame_delta = member_experiment.get_timepoint(i).all_speeds(member_experiment.pixels_per_mm, min_frames=min_frames, framerate=framerate)
                        speeds = speeds + speed
                        frame_deltas = frame_deltas + frame_delta
                    except IndexError:
                        pass
                weighted_mean_speed = sum(x * y for x, y in zip(speeds, frame_deltas)) / sum(frame_deltas)
                speeds_by_group[group_names[n]].append(weighted_mean_speed)
        return speeds_by_group
    
    def get_Experiment_by_data_folder(self, root_stub):
        for experiment in self.experiments:
            if experiment.imageset.root_stub == root_stub:
                return experiment
    
    def easy_group_maker(self, start, total, pre, n, post):
        return [[pre + str(j) + post for j in range(i, start + total, n)] for i in range(start, start + n)]
        
    def all_tracking_as_dataframe(self, groups=None, group_names=None):
        if groups is not None:
            return pd.concat([self.get_Experiment_by_data_folder(member).tracking_as_dataframe().assign(group=group_name) for group, group_name in zip(groups, group_names) for member in group])
        return pd.concat([expmt.tracking_as_dataframe() for expmt in self.experiments])
    
    def pickle_manager_data(self):
        pass
    
    def paralysis_by_group(self, groups, group_names, max_mm_movement=0.25, melted_output=True):
        def df_cleanup(df):
            df = df.astype('float64')
            df['time'] = df.index
            df['time'] = df['time'].apply(lambda x:  pd.Timestamp(datetime.datetime.strptime(x, '%d:%H:%M:%S')), 1)
            return df
            
        all_timepoints = set()
        if groups is None:
            groups = self.groups
        if group_names is None:
            group_names = self.group_names
        for group in groups:
            for member in group:
                experiment = self.get_Experiment_by_data_folder(member)
                if experiment is not None:
                    all_timepoints |= set(experiment.timepoints.keys())
        all_timepoints = sorted(list(all_timepoints))
        paralysis_by_group = pd.DataFrame(index=all_timepoints, columns=group_names)
        SEM_by_group = paralysis_by_group.copy()
        for group, group_name in zip(groups, group_names):
            paralysis_in_group = pd.DataFrame()
            for member in group:
                member_experiment = self.get_Experiment_by_data_folder(member)
                if member_experiment:
                    member_paralysis = member_experiment.paralysis_between_timepoints(max_mm_movement=max_mm_movement)
                    # print(member_paralysis)
                    paralysis_in_group = pd.concat([paralysis_in_group, member_paralysis], axis=1, sort=True)
                    # print(paralysis_in_group)
            # paralysis_by_group = pd.concat([paralysis_by_group, paralysis_in_group.mean(axis=1).rename(group_name)], axis=1)
            paralysis_by_group.loc[:,group_name] = paralysis_in_group.mean(axis=1).rename(group_name)
            SEM_by_group.loc[:,group_name] = paralysis_in_group.std(axis=1).rename(group_name) / math.sqrt(len(group))
        paralysis_by_group = df_cleanup(paralysis_by_group)
        SEM_by_group = df_cleanup(SEM_by_group)
        if melted_output:
            return self.concat_and_organize_mean_and_sem(paralysis_by_group, SEM_by_group)
        return paralysis_by_group, SEM_by_group
    
    def concat_and_organize_mean_and_sem(self, mean, sem):
        mean_melted = mean.melt(id_vars=['time'])
        sem_melted = sem.melt(id_vars=['time'])
        mean_sem_by_group = pd.concat([mean_melted, sem_melted['value']], axis=1)
        mean_sem_by_group.columns = ['time','strain', 'mean', 'sem']
        return mean_sem_by_group

        
    def draw_bounding_boxes_on_all_experiments(self, color_by=None):
        [expmt.draw_bounding_boxes(color_by=color_by) for expmt in self.experiments]

class Experiment():
    '''
    This is an organizatinal level analogous to a folder with images inside.
    An experiment consists of one or more timepoints.
    '''
    def __init__(self, imageset, file_type):
        self.imageset = imageset        # class: IO_tools.ImageDataset
        self.file_type = file_type
        self.inference_results = None
        self.tracked = False
        self.timepoints = {}
        self.pixels_per_mm = None
    
    def __repr__(self):
        if self.inference_results:
            if self.tracked:
                qual = 'tracked'
            else:
                qual = 'infered, untracked'
        else:
            qual = 'uninfered'
        return ' '.join([str(len(self.imageset)), qual, self.file_type, 'files in', self.imageset.root])
    
    def __getitem__(self, n):
        return self.timepoints[sorted(list(self.timepoints.keys()))[n]]
    
    def __len__(self):
        return len(self.timepoints)
            
    def set_calibration(self, pixels_per_mm):
        self.pixels_per_mm = pixels_per_mm
    
    def mm_to_px(self, mm):
        return mm * self.pixels_per_mm
    
    def store_inference_results(self, inference_results):
        self.inference_results = inference_results
    
    def organize_inference_results(self, ts_fmt, min_score=0.9):
        ts_regex_str = ts_fmt.replace(r'%d', r'\d{2}').replace(r'%H', r'\d{2}').replace(r'%M', r'\d{2}').replace(r'%S', r'\d{2}')
        regex_overlappable = '(?=(' + ts_regex_str + '))'
        for img, result in self.inference_results.items():
            filebase = os.path.basename(img).replace(self.file_type, '')
            img_N = filebase.split('_')[-1]
            name_no_serial = '_'.join(filebase.split('_')[0:-1])
            timestamp_from_name = re.findall(regex_overlappable, name_no_serial)[-1]
            tp = datetime.datetime.strptime(timestamp_from_name, ts_fmt).strftime('%d:%H:%M:%S')
            self.result_to_timepoint(tp, img_N, img, result, min_score)
        print('Organized '+str(len(self.inference_results))+' results into '+str(len(self.timepoints))+' timepoints.')
            
    def result_to_timepoint(self, timepoint, n, img, result, min_score=0.9):
        '''
        create the timepoint if necessary, and add the result to it
        '''
        if timepoint not in self.timepoints:
            self.timepoints[timepoint] = Timepoint()
        self.timepoints[timepoint].add_result(n, img, result, min_score=min_score)
        
    def track_worms_within_timepoints(self, pixels_per_mm=None, framerate=2, missing_frames_forgiven=0, max_mm_per_s=2, frame_window=None):
        print(self.imageset.root_stub)
        self.set_framerate(framerate)
        if pixels_per_mm is None:
            pixels_per_mm = self.pixels_per_mm
        [TP.track_worms_across_frames(pixels_per_mm=pixels_per_mm, 
                                      framerate=framerate, 
                                      missing_frames_forgiven=missing_frames_forgiven, 
                                      max_mm_per_s=max_mm_per_s,
                                      frame_window=frame_window) 
                                        for TP in self.timepoints.values()]
    
    @property
    def mean_speeds(self):
        return [TP.mean_speed(pixels_per_mm=self.pixels_per_mm) for TP in self.timepoints.values()]
    
    def draw_bounding_boxes(self, color_by='worm'):
        [TP.draw_bounding_boxes_on_images(self.imageset, color_by) for TP in self.timepoints.values()]
    
    def get_timepoint(self, n):
        return list(self.timepoints.values())[n]
    
    def paralysis_between_timepoints(self, max_mm_movement=0.25):
        '''
        extract 1st imgs and CoMs
        get and apply transforms
        each com, min dist to com in next frame. assign paralyzed
        '''
        results = []
        imgs = []
        sorted_times = sorted(list(self.timepoints.keys()))
        paralysis = pd.DataFrame(index=sorted_times, columns=['paralysis'])
        for time in sorted_times:
            i, r = self.timepoints[time][0]
            imgs.append(i)
            results.append(r['CoM'])
        
        ### TO DO: apply transforms here
        
        for i in range(len(results)-1):
            paralyzed = []
            frame_i = results[i]
            frame_i_plus_1 = results[i+1]
            min_dist = []
            for CoM in frame_i:
                paralyzed.append(min([dist(CoM, CoM_in_next_image) for CoM_in_next_image in frame_i_plus_1]) < (max_mm_movement * self.pixels_per_mm))
            paralysis.loc[sorted_times[i], 'paralysis'] = (sum(paralyzed) / len(paralyzed))
        return paralysis
    
    # def paralysis_between_timepoints(self, max_mm_movement=0.25):
    #     sorted_times = sorted(list(self.timepoints.keys()))
    #     results = pd.DataFrame(columns=['time','img', 'BBox', 'paralyzed'])
    #     for time in sorted_times:
    #         i, r = self.timepoints[time][0]
    #         pd.concat([results, pd.DataFrame({'time':time,'img':i, 'BBox':r['CoM']})])

    
    def draw_paralysis_bounding_boxes(self, max_mm_movement=0.25):
            
        tp_data = {}
        for frame in range(len(self)):
            img_path, _ = self[frame]
            image = imageset.get_by_name_as_gray_Image(img_path)
            image = autocontrast(invert(image), cutoff=0.8, ignore=None)
            image = np.expand_dims(np.array(image), 0)
            tp_data[frame] = {'image': image}
            tp_data[frame]['img_path'] = img_path
            tp_data[frame]['boxes'] = []
            tp_data[frame]['classes'] = []
            tp_data[frame]['scores'] = []
        for worm_number, worm in enumerate(self.list_of_worms):
            for frame, box in worm.bboxes.items():
                bbox = box[[1,0,3,2]]
                tp_data[frame]['boxes'].append(bbox)
                tp_data[frame]['classes'].append(worm_number)
                tp_data[frame]['scores'].append(1)
        category_index = {k:k for k in range(len(self.list_of_worms))}    
        class_to_color_map = {k:random_color() for k in range(len(self.list_of_worms))}
        
        for frame in tp_data.values():
            annotated_image = vis_tools.draw_bounding_boxes_on_image_tensors(frame['image'],
                                                     frame['boxes'],
                                                     frame['classes'],
                                                     frame['scores'],
                                                     category_index,
                                                     max_boxes_to_draw=100,
                                                     min_score_thresh=0.2,
                                                     line_thickness=2,
                                                     skip_scores=True,
                                                     skip_labels=True,
                                                     use_normalized_coordinates=False,
                                                     class_to_color_map=class_to_color_map)
            w_tracking = os.path.join(os.path.split(frame['img_path'])[0], 'w_tracking')
            utils.mkdir(w_tracking)
            Image.fromarray(annotated_image.numpy()).save(os.path.join(w_tracking, os.path.basename(frame['img_path']).replace('.tif', '.jpeg')))

    def set_framerate(self, framerate):
        self.framerate = framerate
        [t.set_framerate(framerate) for t in self.timepoints.values()]
    
    def remove_miscalled_BBox(self, img_title, box_score, and_all_within_mm, autosave=False):
        for tp in self.timepoints.values():
            for img in tp:
                # print(img[0])
                if img[0] == img_title:
                    for score, CoM in zip(img[1]['scores'], img[1]['CoM']):
                        if str(box_score) in str(score):   # convert to string and do check in becasue comparing float32 would be a bitch
                            print('Removed', self._remove_miscalled_BBox(CoM, and_all_within_mm=and_all_within_mm), 'boxes near', CoM)
                            
        if autosave:
            self.save_as_PASCAL_VOC()
    
    def _remove_miscalled_BBox(self, CoM, and_all_within_mm):
        return sum([TP.remove_miscalled_BBox(CoM, and_all_within_px=(and_all_within_mm * self.pixels_per_mm)) for TP in self.timepoints.values()])
        
        
    def save_as_PASCAL_VOC(self):
        for timepoint in self:
            for img_path, results in timepoint:
                IO_tools.saveLabels(img_path, results)
    
    def tracking_as_dataframe(self):
        return pd.concat([TP.as_dataframe().assign(timestamp=timestamp, experiment=self.imageset.root_stub) for timestamp, TP in self.timepoints.items()])
        
    def _track_worms(self, dataset, pixels_per_mm=61, framerate=2, missing_frames_forgiven=0, max_mm_per_s=2):
        # if frame_window is None:
        #     frame_window = (0, len(self)-1)
        # else:
        #     frame_window = (max(frame_window[0], 0), min(frame_window[1], len(self)-1))
        # self.frame_window = frame_window
        # # frame_window[0]
        list_of_worms = [[(0, worm_number)] for worm_number in range(len(dataset[0]['CoM']))]
        for i in range(len(dataset)-1):
            # print('*******************    NEW FRAME   *******************')
            CoMs2 = dataset[i+1]['CoM']
            
            # find closest CoM in next frame
            for one_worm_as_a_list in list_of_worms:
                last_seen_t, last_seen_i = one_worm_as_a_list[-1]      ## each worm is a list of tuples: [(frame, index within CoM[frame] list), ...]
                if (i+1 - last_seen_t) > (missing_frames_forgiven + 1):  ## if worm was last seen too long ago, move on to the next
                    continue 
                CoM = dataset[last_seen_t]['CoM'][last_seen_i]
                distances = [dist(CoM, CoM_in_next_image) for CoM_in_next_image in CoMs2]
                speeds = [(d / pixels_per_mm) * ((i+1 - last_seen_t) / framerate) for d in distances]
                if len(speeds) == 0:
                    continue
                if min(speeds) > max_mm_per_s:
                    continue
                one_worm_as_a_list.append((i + 1, speeds.index(min(speeds))))
            
            # check if any worms merged
            CoMs_logged_so_far_in_this_frame = {}   # Will be {this_frame_CoM: worm_i}
            for worm_i, one_worm_as_a_list in enumerate(list_of_worms):
                # print(worm_i, one_worm_as_a_list)
                if one_worm_as_a_list[-1][0] == (i + 1):  # if worm in most recent frame. frame is 1-based, but i is 0-based, so I have do i+1
                    # print('in recent frame')
                    this_CoM = tuple(dataset[i+1]['CoM'][one_worm_as_a_list[-1][1]])  #  CoM of worm in frame i+1 (the most recent frame). Must convert to a tuple so its hashable for the dict. 
                    if this_CoM in CoMs_logged_so_far_in_this_frame:
                        #### What i think I want to happen is keep the closest worm to this_CoM.
                        #### So, compare dist(this_CoM, worm1_previous_CoM) to dist(this_CoM, worm2_previous_CoM)
                        worm1_last_seen_in_frame = one_worm_as_a_list[-2][0]
                        worm1_last_seen_CoM = one_worm_as_a_list[-2][1]
                        this_dist_to_next_frame = dist(this_CoM, dataset[worm1_last_seen_in_frame]['CoM'][worm1_last_seen_CoM])
                        
                        worm2_last_seen_in_frame = list_of_worms[CoMs_logged_so_far_in_this_frame[this_CoM]][-2][0]
                        worm2_last_seen_CoM = list_of_worms[CoMs_logged_so_far_in_this_frame[this_CoM]][-2][1]
                        other_worm_dist         = dist(this_CoM, dataset[worm2_last_seen_in_frame]['CoM'][worm2_last_seen_CoM])
                        
                        if this_dist_to_next_frame < other_worm_dist:
                            list_of_worms[CoMs_logged_so_far_in_this_frame[this_CoM]].pop(-1)
                            CoMs_logged_so_far_in_this_frame[this_CoM] = worm_i
                        else:
                            one_worm_as_a_list.pop(-1)
                    else:
                        CoMs_logged_so_far_in_this_frame[this_CoM] = worm_i
            
            # Start new worms from unused CoMs
            indexes_to_skip = []
            for one_worm_as_a_list in list_of_worms:
                if one_worm_as_a_list[-1][0] == (i + 1):
                    indexes_to_skip.append(one_worm_as_a_list[-1][1])
            for j in range(len(CoMs2)):
                if j in indexes_to_skip:
                    continue
                list_of_worms.append([(i + 1, j)])
        
        return list_of_worms
    
    def _track_worms_across_TP_frames(self, pixels_per_mm=None, framerate=2, missing_frames_forgiven=0, max_mm_per_s=2, frame_window=None):
        print(self.imageset.root_stub)
        self.set_framerate(framerate)
        if pixels_per_mm is None:
            pixels_per_mm = self.pixels_per_mm
        for TP in self.timepoints.values():
            # TP.track_worms_across_frames(pixels_per_mm=pixels_per_mm, 
            #                           framerate=framerate, 
            #                           missing_frames_forgiven=missing_frames_forgiven, 
            #                           max_mm_per_s=max_mm_per_s,
            #                           frame_window=frame_window) 
            
            if frame_window is None:
                frame_window = (0, len(TP)-1)
            else:
                frame_window = (max(frame_window[0], 0), min(frame_window[1], len(TP)-1))
            TP.frame_window = frame_window
            
            dataset = [TP[i][1] for i in range(frame_window[0], frame_window[1])]
            
            list_of_worms = self._track_worms(dataset, 
                pixels_per_mm=pixels_per_mm, 
                framerate=framerate, 
                missing_frames_forgiven=missing_frames_forgiven, 
                max_mm_per_s=max_mm_per_s)
            TP.list_of_worms = [Worm(one_worm_as_a_list, TP, ID) for ID, one_worm_as_a_list in enumerate(list_of_worms)]
            TP.save_worms_to_results()
    
    def identify_debris(self, missing_frames_forgiven=0, max_mm_per_step=0.1, pixels_per_mm=None, min_frames=None):
        if pixels_per_mm is None:
            pixels_per_mm = self.pixels_per_mm
        dataset = [TP[0][1] for TP in self.timepoints.values()]
        # print(dataset)
        if min_frames is None:
            min_frames = len(dataset)
        list_of_debris = [d for d in self._track_worms(dataset, 
            pixels_per_mm=pixels_per_mm, 
            framerate=1, 
            missing_frames_forgiven=missing_frames_forgiven, 
            max_mm_per_s=max_mm_per_step)
            if len(d) >= min_frames]
        # print(list_of_debris)
        print('Identified ' + str(len(list_of_debris)) + ' particles of debris')
        return [np.mean(np.array([dataset[p[0]]['CoM'][p[1]] for p in d]), axis=0) for d in list_of_debris]
    
    def remove_debris(self, missing_frames_forgiven=0, max_mm_per_step=0.1, pixels_per_mm=None, min_frames=None, remove_within_mm=0.2):
        mean_debris_positions = self.identify_debris(missing_frames_forgiven=missing_frames_forgiven,
            max_mm_per_step=max_mm_per_step,
            pixels_per_mm=pixels_per_mm,
            min_frames=min_frames)
        # print(mean_debris_positions)
        for TP in self.timepoints.values():
            for frame in range(len(TP)):
                remove_i = []
                for i in range(len(TP[frame][1]['CoM'])):
                    for d in mean_debris_positions:
                        if dist(d, TP[frame][1]['CoM'][i]) <  self.mm_to_px(remove_within_mm):
                            remove_i.append(i)
                if remove_i:
                    for key in TP[frame][1]:
                        TP[frame][1][key] = np.delete(TP[frame][1][key], remove_i, 0)
            
    
class Timepoint():
    '''
    hold all data for a timepoint/imagetrain
    Track speed within the train
    '''
    def __init__(self):
        self.results = {}
        self.images = {}
        self.iter_n = -1
        self.n = []
    
    def __getitem__(self, n):
        return self.images[self.n[n]], self.results[self.n[n]]
    
    def __len__(self):
        return len(self.results)
    
    def __iter__(self):
        return self
        
    def __next__(self):
        self.iter_n = self.iter_n + 1
        if self.iter_n < len(self):
            return self.__getitem__(self.iter_n)
        self.iter_n = -1  
        raise StopIteration
    
    @property
    def frame_window_length(self):
        return self.frame_window[1]-self.frame_window[0]+1
    
    def set_framerate(self, framerate):
        self.framerate = framerate
        
    def add_result(self, n, img, result, min_score=0.9):
        n = int(n)
        scores = result['scores']
        for cat in result:
            result[cat] = np.array([each for each, score in zip(result[cat], scores) if score>=min_score], dtype=result[cat].dtype)
        self.results[n] = result
        self.images[n] = img
        self.n = sorted(list(self.results.keys()))
        

    
      
    
        
            
    def track_worms_across_frames(self, pixels_per_mm=61, framerate=2, missing_frames_forgiven=0, max_mm_per_s=2, frame_window=None):
        if frame_window is None:
            frame_window = (0, len(self)-1)
        else:
            frame_window = (max(frame_window[0], 0), min(frame_window[1], len(self)-1))
        self.frame_window = frame_window
        # frame_window[0]
        list_of_worms = [[(frame_window[0], idx)] for idx in range(len(self[frame_window[0]][1]['CoM']))]
        for i in range(frame_window[0], frame_window[1]):
            # print('*******************    NEW FRAME   *******************')
            CoMs2 = self[i+1][1]['CoM']
            
            # find closest CoM in next frame
            for one_worm_as_a_list in list_of_worms:
                last_seen_t, last_seen_i = one_worm_as_a_list[-1]      ## each worm is a list of tuples: [(frame, index within CoM[frame] list), ...]
                if (i+1 - last_seen_t) > (missing_frames_forgiven + 1):  ## if worm was last seen too long ago, move on to the next
                    continue 
                CoM = self[last_seen_t][1]['CoM'][last_seen_i]
                distances = [dist(CoM, CoM_in_next_image) for CoM_in_next_image in CoMs2]
                speeds = [(d / pixels_per_mm) * ((i+1 - last_seen_t) / framerate) for d in distances]
                if len(speeds) == 0:
                    continue
                if min(speeds) > max_mm_per_s:
                    continue
                one_worm_as_a_list.append((i + 1, speeds.index(min(speeds))))
            
            # check if any worms merged
            CoMs_logged_so_far_in_this_frame = {}   # Will be {this_frame_CoM: worm_i}
            for worm_i, one_worm_as_a_list in enumerate(list_of_worms):
                # print(worm_i, one_worm_as_a_list)
                if one_worm_as_a_list[-1][0] == (i + 1):  # if worm in most recent frame. frame is 1-based, but i is 0-based, so I have do i+1
                    # print('in recent frame')
                    this_CoM = tuple(self[i+1][1]['CoM'][one_worm_as_a_list[-1][1]])  #  CoM of worm in frame i+1 (the most recent frame). Must convert to a tuple so its hashable for the dict. 
                    if this_CoM in CoMs_logged_so_far_in_this_frame:
                        #### What i think I want to happen is keep the closest worm to this_CoM.
                        #### So, compare dist(this_CoM, worm1_previous_CoM) to dist(this_CoM, worm2_previous_CoM)
                        worm1_last_seen_in_frame = one_worm_as_a_list[-2][0]
                        worm1_last_seen_CoM = one_worm_as_a_list[-2][1]
                        this_dist_to_next_frame = dist(this_CoM, self[worm1_last_seen_in_frame][1]['CoM'][worm1_last_seen_CoM])
                        
                        worm2_last_seen_in_frame = list_of_worms[CoMs_logged_so_far_in_this_frame[this_CoM]][-2][0]
                        worm2_last_seen_CoM = list_of_worms[CoMs_logged_so_far_in_this_frame[this_CoM]][-2][1]
                        other_worm_dist         = dist(this_CoM, self[worm2_last_seen_in_frame][1]['CoM'][worm2_last_seen_CoM])
                        
                        if this_dist_to_next_frame < other_worm_dist:
                            list_of_worms[CoMs_logged_so_far_in_this_frame[this_CoM]].pop(-1)
                            CoMs_logged_so_far_in_this_frame[this_CoM] = worm_i
                        else:
                            one_worm_as_a_list.pop(-1)
                    else:
                        CoMs_logged_so_far_in_this_frame[this_CoM] = worm_i
            
            # Start new worms from unused CoMs
            indexes_to_skip = []
            for one_worm_as_a_list in list_of_worms:
                if one_worm_as_a_list[-1][0] == (i + 1):
                    indexes_to_skip.append(one_worm_as_a_list[-1][1])
            for j in range(len(CoMs2)):
                if j in indexes_to_skip:
                    continue
                list_of_worms.append([(i + 1, j)])
        
        self.list_of_worms = [Worm(one_worm_as_a_list, self, ID) for ID, one_worm_as_a_list in enumerate(list_of_worms)]
        self.save_worms_to_results()
    
    def save_worms_to_results(self):
        # WormID = -1 means no worm
        for result in self.results.values():
            result['wormID'] = np.array([-1] * len(result['labels']), dtype='int')
        for worm in self.list_of_worms:
            for frame, idx in worm.worm_as_list:
                self[frame][1]['wormID'][idx] = worm.ID
                        
    def calc_speed_of_worms(self, pixels_per_mm=61, framerate=2):
        for worm in self.list_of_worms:
            worm.calculate_speed(pixels_per_mm=pixels_per_mm, framerate=framerate)
    
    def mean_speed(self, min_frames=2, pixels_per_mm=61, framerate=2):
        '''
        calculate mean speed of worms, weighted by the time length of the track
        '''
        speeds, frame_deltas = self.all_speeds(pixels_per_mm=pixels_per_mm, min_frames=min_frames, framerate=framerate)
        if speeds:
            return sum(x * y for x, y in zip(speeds, frame_deltas)) / sum(frame_deltas)
    
    def all_speeds(self, pixels_per_mm, min_frames=2, framerate=2):
        self.calc_speed_of_worms(pixels_per_mm=pixels_per_mm, framerate=framerate)
        speeds = [worm.mean_speed for worm in self.list_of_worms if worm.N_frames >= min_frames]
        frame_deltas = [worm.delta_T for worm in self.list_of_worms if worm.N_frames >= min_frames]
        return speeds, frame_deltas

    def draw_bounding_boxes_on_images(self, imageset, color_by):
        if color_by is 'worm':
            self.draw_bounding_boxes_by_worm(imageset)
        if color_by is None:
            self.draw_bounding_boxes(imageset)
    
    def draw_bounding_boxes(self, imageset):
        tp_data = {}
        for frame in range(len(self)):
            img_path, results = self[frame]
            image = imageset.get_by_name_as_gray_Image(img_path)
            image = autocontrast((image), cutoff=0.8, ignore=None)
            image = np.expand_dims(np.array(image), 0)
            tp_data[frame] = {'image': image}
            tp_data[frame]['img_path'] = img_path
            tp_data[frame]['boxes'] = [box[[1,0,3,2]] for box in results['boxes']]
            tp_data[frame]['classes'] = [1 for _ in results['labels']]
            tp_data[frame]['scores'] = [1 for _ in results['scores']]
        category_index = {1: 1}    
        class_to_color_map = {1: 'black'}
        
        for frame in tp_data.values():
            annotated_image = vis_tools.draw_bounding_boxes_on_image_tensors(frame['image'],
                                                     frame['boxes'],
                                                     frame['classes'],
                                                     frame['scores'],
                                                     category_index,
                                                     max_boxes_to_draw=100,
                                                     min_score_thresh=0.2,
                                                     line_thickness=2,
                                                     skip_scores=True,
                                                     skip_labels=True,
                                                     use_normalized_coordinates=False,
                                                     class_to_color_map=class_to_color_map)
            w_tracking = os.path.join(os.path.split(frame['img_path'])[0], 'w_tracking')
            utils.mkdir(w_tracking)
            Image.fromarray(annotated_image.numpy()).save(os.path.join(w_tracking, os.path.basename(frame['img_path']).replace('.tif', '.jpeg')))

    
    def draw_bounding_boxes_by_worm(self, imageset):
        def random_color():
            color = tuple(np.random.choice(range(256), size=3))
            while (50 > sum(list(color)) > 700) or (sum(np.abs(np.roll(color, 1) - color)) < 200):  ## If too dark, too bright, or too gray
                color = random_color()
            return color
            
        tp_data = {}
        for frame in range(len(self)):
            img_path, _ = self[frame]
            image = imageset.get_by_name_as_gray_Image(img_path)
            image = autocontrast(invert(image), cutoff=0.8, ignore=None)
            image = np.expand_dims(np.array(image), 0)
            tp_data[frame] = {'image': image}
            tp_data[frame]['img_path'] = img_path
            tp_data[frame]['boxes'] = []
            tp_data[frame]['classes'] = []
            tp_data[frame]['scores'] = []
        for worm_number, worm in enumerate(self.list_of_worms):
            for frame, box in worm.bboxes.items():
                bbox = box[[1,0,3,2]]
                tp_data[frame]['boxes'].append(bbox)
                tp_data[frame]['classes'].append(worm_number)
                tp_data[frame]['scores'].append(1)
        category_index = {k:k for k in range(len(self.list_of_worms))}    
        class_to_color_map = {k:random_color() for k in range(len(self.list_of_worms))}
        
        for frame in tp_data.values():
            annotated_image = vis_tools.draw_bounding_boxes_on_image_tensors(frame['image'],
                                                     frame['boxes'],
                                                     frame['classes'],
                                                     frame['scores'],
                                                     category_index,
                                                     max_boxes_to_draw=100,
                                                     min_score_thresh=0.2,
                                                     line_thickness=2,
                                                     skip_scores=True,
                                                     skip_labels=True,
                                                     use_normalized_coordinates=False,
                                                     class_to_color_map=class_to_color_map)
            w_tracking = os.path.join(os.path.split(frame['img_path'])[0], 'w_tracking')
            utils.mkdir(w_tracking)
            Image.fromarray(annotated_image.numpy()).save(os.path.join(w_tracking, os.path.basename(frame['img_path']).replace('.tif', '.jpeg')))
    
    def remove_miscalled_BBox(self, CoM_1, and_all_within_px):
        n_removed = 0
        for i in range(len(self)):
            list_of_CoMs = self[i][1]['CoM']
            results = self[i][1]
            for cat in results:
                results[cat] = np.array([each for each, CoM_2 in zip(results[cat], list_of_CoMs) if (dist(CoM_1, CoM_2) > and_all_within_px)], dtype=results[cat].dtype)
            n_removed = n_removed + len(list_of_CoMs) - len(results[cat])
        return n_removed
    
    def as_dataframe(self):
        def to_1d(arr):
            if len(arr.shape) == 1:
                return arr
            else:
                return np.apply_along_axis(lambda x: str(x), 1, arr)
                
        return pd.DataFrame({key: to_1d(self.results[n][key]) for n in self.n for key in self.results[n]})
            
                
                

class Worm():
    def __init__(self, worm_as_list, TP, ID, pixels_per_mm=61, framerate=2):
        self.coords = {}
        self.bboxes = {}
        self.ID = ID
        self.worm_as_list = worm_as_list
        for frame, idx in worm_as_list:
            self.coords[frame] = TP[frame][1]['CoM'][idx]
            self.bboxes[frame] = TP[frame][1]['boxes'][idx]
        self.frames = sorted(list(self.coords.keys()))
        if len(self.frames) > 1:
            self.calculate_speed(pixels_per_mm=pixels_per_mm, framerate=framerate)
    
    def calculate_speed(self, pixels_per_mm=61, framerate=2):
        self.speeds = {}
        for i in range(len(self.coords)-1):
            self.speeds[self.frames[i]] = (dist(self.coords[self.frames[i]], self.coords[self.frames[i+1]]) / pixels_per_mm) * (framerate / (self.frames[i+1] - self.frames[i]))
    
    @property
    def mean_speed(self):
        speeds = list(self.speeds.values())
        frame_deltas = [(self.frames[i+1] - self.frames[i]) for i in range(len(self.coords)-1)]
        if speeds:
            return sum(x * y for x, y in zip(speeds, frame_deltas)) / sum(frame_deltas)
    
    def total_distance(self, pixels_per_mm=61):
        return sum([(dist(self.coords[self.frames[i]], self.coords[self.frames[i+1]]) / pixels_per_mm) for i in range(len(self.coords)-1)])
    
    @property
    def delta_T(self):
        return self.frames[-1] - self.frames[0]
        
    @property
    def N_frames(self):
        return len(self.frames)


def get_center_of_worm(cropped_image):
    pix = np.array(cropped_image)
    threshold = skimage.filters.threshold_triangle(pix) 
    threshed_img = (pix-threshold) * (pix>threshold)
    center_of_mass = ndimage.measurements.center_of_mass(threshed_img)
    center_of_pix = ndimage.measurements.center_of_mass(threshed_img>0)
    return center_of_mass, center_of_pix


def dist(pt1, pt2):
    if (pt1 is None) or (pt2 is None):
        return None
    delta_x = pt1[0]-pt2[0]
    delta_y = pt1[1]-pt2[1]
    return math.sqrt(delta_x**2 + delta_y**2)

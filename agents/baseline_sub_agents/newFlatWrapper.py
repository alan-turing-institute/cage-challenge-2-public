from datetime import datetime

from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Shared import Observation
from CybORG.Shared.Actions import ShellSleep
from CybORG.Shared.Enums import OperatingSystemType, SessionType, ProcessName, Path, ProcessType, ProcessVersion, \
    AppProtocol, FileType, ProcessState, Vulnerability, Vendor, PasswordHashType, BuiltInGroups, \
    OperatingSystemDistribution, OperatingSystemVersion, OperatingSystemKernelVersion, Architecture, \
    OperatingSystemPatch, FileVersion

import inspect, random
from pprint import pprint
import pickle
import os 
import numpy as np
import copy
import sys
from csv import writer

class NewFixedFlatWrapper(BaseWrapper):
    def __init__(self, env: BaseWrapper=None, agent=None, log_name='flat.txt'):
        super().__init__(env, agent)
        self.MAX_HOSTS = 5 # changed 
        self.MAX_PROCESSES = 11 #100
        self.MAX_CONNECTIONS = 1
        self.MAX_VULNERABILITIES = 1
        self.MAX_INTERFACES = 1 # 4
        self.MAX_FILES = 2 # 10
        self.MAX_SESSIONS = 0 # 2
        self.MAX_USERS = 0 # 5
        self.MAX_GROUPS = 0 #1
        self.MAX_PATCHES = 0 # 10
        self.hostname = {'Defender': 1, 'Enterprise0': 2, 'Enterprise1': 3, 'Enterprise2': 4, \
                         'Op_Host0': 5, 'Op_Host1': 6, 'Op_Host2': 7, 'Op_Server0': 8, 'User0': 9, 'User1': 10, \
                         'User2': 11, 'User3': 12, 'User4': 13}
        self.username = {}
        self.group_name = {}
        self.process_name = {}
        self.interface_name = {}
        self.path = {}
        self.password = {}
        self.password_hash = {}
        self.file = {}
        # added 
        self.session_pid = {}
        self.process_pid = {}
        self.process_ppid = {}
        # self.subnet_netadd = {}
        # self.subnet_prefix = {}
        self.ipadd = {}
        self.localadd = {}
        self.remoteadd = {}
        self.remote_port = {}
        self.local_port = {}

        self.count = 0
        self.log_name = log_name

    def get_action(self, observation, action_space):

        action = self.agent.get_action(self.observation_change(observation), self.action_space_change(action_space))

        action_class = action['action']
        params = {}
        for p in inspect.signature(action_class).parameters:
            if p in action:
                params[p] = action[p]
            else:
                action_class = ShellSleep
                params = {}
                break
        action = action_class(**params)
        return action

    # def action_space_change(self, action_space: dict) -> dict:
    #     action_space.pop('process')
    #     action_space['session'] = {0: True}
    #     action_space['username'] = {'pi': action_space['username']['pi'],
    #                                 'vagrant': action_space['username']['vagrant']}
    #     action_space['password'] = {'raspberry': action_space['password']['raspberry'],
    #                                 'vagrant': action_space['password']['vagrant']}
    #     action_space['port'] = {22: action_space['port'][22]}
    #     return action_space

    def observation_change(self, obs: dict) -> list:
        numeric_obs = copy.deepcopy(obs)
        flat_obs = []
        hostid = str(random.randint(0, self.MAX_HOSTS+1))
  
        orig_stdout = sys.stdout
        f = open('obs_dict.txt', 'a')
        sys.stdout = f
        self.count +=1
        if len(obs.keys()) != 14 and self.count%2==0:
            print(f'{self.count/2}-----------')
            pprint(obs)
        sys.stdout = orig_stdout
        f.close()


        while len(numeric_obs) < self.MAX_HOSTS:
            hostid = str(random.randint(0, self.MAX_HOSTS+1))
            if hostid not in numeric_obs.keys():
                numeric_obs[hostid] = {}

        if len(numeric_obs) > self.MAX_HOSTS:
            with open(self.log_name, 'a') as file:
                file.write(f"MAX_HOSTS: {len(numeric_obs)}\n")
        while len(numeric_obs) > self.MAX_HOSTS:
            numeric_obs.popitem()
            # print('numeric_obs', numeric_obs.keys())
            # raise ValueError("Too many hosts in observation for fixed size")

        # print('numeric_obs.items()', numeric_obs.keys())
        for key_name, host in numeric_obs.items():
            # print('====== ', key_name)
            if key_name == 'success':
                # flat_obs.append(float(host.value)/3) # @@@ ??? not sure why divided by 3 == maybe there are only three types of success status
                flat_obs += list(np.eye(4)[int(host.value)]) # make it as one hot
                # flat_obs += list(np.eye(4)[int(host)])
            elif not isinstance(host, dict):
                raise ValueError('Host data must be a dict')
            else:
                if key_name in self.hostname:
                    # self.hostname[key_name] = len(self.hostname) # @@@ "user 3": 0 (value is the order ofh)
                    # print(self.hostname.keys())
                    flat_obs += list(np.eye(13)[self.hostname[key_name]-1]) # one-hot representation
                else:
                    # with open(self.log_name, 'a') as file:
                    #     file.write("@@@@ no hostname info??")
                    # flat_obs.append(-1.0)
                    flat_obs += 13*[-1.0]
                print('hostname', len(flat_obs)-1)
                # if 'System info' in host:
                    # if "Hostname" in host["System info"]:
                    #     element = host["System info"]["Hostname"]
                    #     if element not in self.hostname:
                    #         self.hostname[element] = len(self.hostname) # @@@ "user 3": 0 (value is the order ofh)
                    #     # element = self.hostname[element]/self.MAX_HOSTS # normalise it
                    #     # flat_obs.append(float(element))
                    #     flat_obs += list(np.eye(14)[self.hostname[element]]) # one-hot representation
                    # else:
                    #     with open(self.log_name, 'a') as file:
                    #         file.write("@@@@ no hostname info??")
                    #     # flat_obs.append(-1.0)
                    #     flat_obs += 14*[-1.0]
                        
                #     if "OSType" in host["System info"]:
                #         if host["System info"]["OSType"] != -1:
                #             element = host["System info"]["OSType"].value/len(OperatingSystemType.__members__)
                #         else:
                #             element = -1
                        
                #         flat_obs.append(float(element))
                #     else:
                #         flat_obs.append(-1.0)

                #     if "OSDistribution" in host["System info"]:
                #         if host["System info"]["OSDistribution"] != -1:
                #             element = host["System info"]["OSDistribution"].value / len(OperatingSystemDistribution.__members__)
                #         else:
                #             element = -1
                        
                #         flat_obs.append(float(element))
                #     else:
                #         flat_obs.append(-1.0)

                #     if "OSVersion" in host["System info"]:
                #         if host["System info"]["OSVersion"] != -1:
                #             element = host["System info"]["OSVersion"].value / len(OperatingSystemVersion.__members__)
                #         else:
                #             element = -1
                        
                #         flat_obs.append(float(element))
                #     else:
                #         flat_obs.append(-1.0)

                #     # if "OSKernelVersion" in host["System info"]:
                #     #     if host["System info"]["OSKernelVersion"] != -1:
                #     #         element = host["System info"]["OSKernelVersion"].value / len(OperatingSystemKernelVersion.__members__)
                #     #     else:
                #     #         element = -1
                        
                #     #     flat_obs.append(float(element))
                #     # else:
                #     #     flat_obs.append(-1.0)

                #     if "Architecture" in host["System info"]:
                #         if host["System info"]["Architecture"] != -1:
                #             element = host["System info"]["Architecture"].value / len(Architecture.__members__)
                #         else:
                #             element = -1
                        
                #         flat_obs.append(float(element))
                #     else:
                #         flat_obs.append(-1.0)

                #     # if 'Local Time' in host["System info"]:
                #     #     element = (host["System info"]['Local Time'] - datetime(2020, 1, 1)).total_seconds()
                        
                #     #     flat_obs.append(float(element))
                #     # else:
                #     #     flat_obs.append(-1.0)

                #     if "os_patches" not in host["System info"]:
                #         host["System info"]["os_patches"] = []

                #     while len(host["System info"]["os_patches"]) < self.MAX_PATCHES:
                #         host["System info"]["os_patches"].append(-1.0)
                #     if len(host["System info"]["os_patches"]) > self.MAX_PATCHES:
                #         with open(self.log_name, 'a') as file:
                #             file.write(f'MAX_PATCHES, {len(host["System info"]["os_patches"])}\n')
                #         # raise ValueError("Too many processes in observation for fixed size")
                #     for patch_idx, patch in enumerate(host["System info"]["os_patches"]):
                #         if patch != -1:
                #             element = patch.value / len(OperatingSystemPatch.__members__)
                #         else:
                #             element = patch
                        
                #         flat_obs.append(float(element))
                # else:
                #     flat_obs += [-1.0]*4
                #     for num_patches in range(self.MAX_PATCHES):
                #         flat_obs.append(-1.0)

                if 'Processes' not in host:
                    host["Processes"] = []
                while len(host["Processes"]) < self.MAX_PROCESSES:
                    host["Processes"].append({})

                if len(host["Processes"]) > self.MAX_PROCESSES:
                    with open(self.log_name, 'a') as file:
                        file.write(f'MAX_PROCESSES, {len(host["Processes"])}\n')
                while len(host["Processes"]) > self.MAX_PROCESSES:
                    host["Processes"].pop()
                    # raise ValueError("Too many processes in observation for fixed size")

                for proc_idx, process in enumerate(host['Processes']):
                    if "PID" in process:
                        # element = process["PID"]
                        # if element not in self.process_pid:
                        #     self.process_pid[element] = len(self.process_pid)
                            # with open(self.log_name, 'a') as file:
                            #     file.write(f"process_pid, {len(self.process_pid)}\n")
                        # element = self.process_pid[element]
                        # flat_obs.append(float(element))
                        # flat_obs.append(float(process["PID"])/32768)
                        flat_obs.append(float(process["PID"])/1000)
                    else:
                        flat_obs.append(-1.0)
                    print('pid', len(flat_obs)-1)

                    if "PPID" in process:
                        element = process["PPID"]
                        if element not in self.process_ppid:
                            self.process_ppid[element] = len(self.process_ppid)
                            # with open(self.log_name, 'a') as file:
                            #     file.write(f"process_ppid, {len(self.process_ppid)}\n")
                        element = self.process_ppid[element]
                        flat_obs.append(float(element))
                        # flat_obs.append(float(process["PPID"])/32768)
                    else:
                            flat_obs.append(-1.0)
                    print('ppid', len(flat_obs)-1)
                    # if "Process Name" in process:
                    #     element = process["Process Name"]
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"Process Name, {element}\n")
                    #     if element not in self.process_name:
                    #         self.process_name[element] = len(self.process_name)
                    #     element = self.process_name[element]
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    if "Username" in process:
                        element = process["Username"]
                        with open(self.log_name, 'a') as file:
                            file.write(f"Username, {element}\n")
                        if element not in self.username:
                            self.username[element] = len(self.username)
                        element = self.username[element]
                        flat_obs.append(float(element))
                    else:
                        flat_obs.append(-1.0)
                    print('Username', len(flat_obs)-1)
                    # if "Path" in process:
                    #     element = process["Path"]
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"Path, {element}\n")
                    #     if element not in self.path:
                    #         self.path[element] = len(self.path)
                    #     element = self.path[element]
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Known Process" in process:
                    #     if process["Known Process"] != -1:
                    #         element = process["Known Process"].value / len(ProcessName.__members__)
                    #     else:
                    #         element = -1.0
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"Known Process, {element}\n")
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Known Path" in process:
                    #     if process["Known Path"] != -1:
                    #         element = process["Known Path"].value / len(Path.__members__)
                    #     else:
                    #         element = -1.0
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"Known Path, {element}\n")
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Process Type" in process:
                    #     if process["Process Type"] != -1:
                    #         element = process["Process Type"].value / len(ProcessType.__members__)
                    #     else:
                    #         element = -1.0
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"Process Type, {element}\n")
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Process Version" in process:
                    #     if process["Process Version"] != -1:
                    #         element = process["Process Version"].value / len(ProcessVersion.__members__)
                    #     else:
                    #         element = -1.0
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"Process Version, {element}\n")
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    if "Connections" not in process:
                        process["Connections"] = []
                    while len(process["Connections"]) < self.MAX_CONNECTIONS:
                        process["Connections"].append({})
                    if len(process["Connections"]) > self.MAX_CONNECTIONS:
                        with open(self.log_name, 'a') as file: 
                            file.write(f'MAX_CONNECTIONS: {len(process["Connections"])}\n')

                    for conn_idx, connection in enumerate(process["Connections"]):
                        if "local_port" in connection:
                            element = int(connection["local_port"])
                            if element not in self.local_port:
                                self.local_port[element] = len(self.local_port)
                                # with open(self.log_name, 'a') as file:
                                #     file.write(f"local_port, {len(self.local_port)}\n")
                            element = self.local_port[element]
                            flat_obs.append(float(element))
                            # flat_obs.append(float(connection["local_port"])/65535)
                        else:
                            flat_obs.append(-1.0)
                        print('local_port', len(flat_obs)-1)

                        if "remote_port" in connection:
                            element = int(connection["remote_port"])
                            if element not in self.remote_port:
                                self.remote_port[element] = len(self.remote_port)
                                # with open(self.log_name, 'a') as file:
                                #     file.write(f"remote_port, {len(self.remote_port)}\n")
                            element = self.remote_port[element]
                            flat_obs.append(float(element))
                            # flat_obs.append(float(connection["remote_port"])/65535)
                        else:
                            flat_obs.append(-1.0)
                        print('remote_port', len(flat_obs)-1)

                        if "local_address" in connection:
                            element = int(connection["local_address"])
                            if element not in self.localadd:
                                self.localadd[element] = len(self.localadd)
                                # with open(self.log_name, 'a') as file:
                                #     file.write(f"localadd, {len(self.localadd)}\n")
                            element = self.localadd[element]
                            flat_obs.append(float(element))
                            # flat_obs.append(float(element)/4294967296)
                        else:
                            flat_obs.append(-1.0)
                        print('local_address', len(flat_obs)-1)

                        if "remote_address" in connection:
                            element = int(connection["remote_address"])
                            if element not in self.remoteadd:
                                self.remoteadd[element] = len(self.remoteadd)
                                # with open(self.log_name, 'a') as file:
                                #     file.write(f"remoteadd, {len(self.remoteadd)}\n")
                            element = self.remoteadd[element]
                            flat_obs.append(float(element))
                            # flat_obs.append(float(element)/4294967296)
                        else:
                            flat_obs.append(-1.0)
                        print('remote_address', len(flat_obs)-1)

                        # if "Application Protocol" in connection:
                        #     if connection["Application Protocol"] != -1:
                        #         element = connection["Application Protocol"].value / len(AppProtocol.__members__)
                        #     else:
                        #         element = -1.0
                        #     with open(self.log_name, 'a') as file:
                        #         file.write(f"Application Protocol, {element}\n")
                        #     flat_obs.append(float(element))
                        # else:
                        #     flat_obs.append(-1.0)

                        # if "Status" in connection:
                        #     if connection["Status"] != -1:
                        #         element = connection["Status"].value / len(ProcessState.__members__)
                        #     else:
                        #         element = -1.0
                        #     with open(self.log_name, 'a') as file:
                        #         file.write(f"Status, {element}\n")
                        #     flat_obs.append(float(element))
                        # else:
                        #     flat_obs.append(-1.0)

                    if "Vulnerability" in process:
                        if len(process["Vulnerability"]) > self.MAX_VULNERABILITIES:
                            with open(self.log_name, 'a') as file: 
                                file.write(f'MAX_VULNERABILITIES: {len(process["Vulnerability"])}\n')
                        for idx, element in enumerate(process["Vulnerability"]):
                            if element != -1:
                                element = element.value / len(Vulnerability.__members__)
                            with open(self.log_name, 'a') as file:
                                file.write(f"Vulnerability, {element}\n")
                            flat_obs.append(float(element))
                    else:
                        for idx in range(self.MAX_VULNERABILITIES):
                            flat_obs.append(-1.0)
                    print('vulnerability', len(flat_obs)-1)

                if "Files" not in host:
                    host["Files"] = []
                while len(host["Files"]) < self.MAX_FILES:
                    host["Files"].append({})
                
                if len(host["Files"]) > self.MAX_FILES:
                    with open(self.log_name, 'a') as file: 
                        file.write(f'MAX_FILES: {len(host["Files"])}\n')
                while len(host["Files"]) > self.MAX_FILES:
                    host["Files"].pop()
                    # raise ValueError("Too many files in observation for fixed size")
                

                for file_idx, file in enumerate(host['Files']):
                    if "Path" in file:
                        element = file["Path"]
                        with open(self.log_name, 'a') as afile:
                            afile.write(f"file Path, {element}\n")
                        if element not in self.path:
                            self.path[element] = len(self.path)
                        element = self.path[element]
                        flat_obs.append(float(element))
                    else:
                        flat_obs.append(-1.0)
                    print('file Path', len(flat_obs)-1)

                    if "Known Path" in file:
                        if file["Known Path"] != -1:
                            element = file["Known Path"].value / len(Path.__members__)
                        else:
                            element = -1.0
                        with open(self.log_name, 'a') as afile:
                            afile.write(f"file Known Path, {element}\n")
                        flat_obs.append(float(element))
                    else:
                        flat_obs.append(-1.0)
                    print('file Known Path', len(flat_obs)-1)

                    if "File Name" in file:
                        element = file["File Name"]
                        with open(self.log_name, 'a') as afile:
                            afile.write(f"File Name, {element}\n")
                        if element not in self.file:
                            self.file[element] = len(self.file)
                        element = self.file[element]
                        flat_obs.append(float(element))
                    else:
                        flat_obs.append(-1.0)
                    print('file Name', len(flat_obs)-1)


                    if "Known File" in file:
                        if file["Known File"] != -1:
                            element = file["Known File"].value / len(FileType.__members__)
                        else:
                            element = -1.0
                        with open(self.log_name, 'a') as afile:
                            afile.write(f"Known File, {element}\n")
                        flat_obs.append(float(element))
                    else:
                            flat_obs.append(-1.0)
                    print('Known file', len(flat_obs)-1)

                    # if "Type" in file:
                    #     if file["Type"] != -1:
                    #         element = file["Type"].value / len(FileType.__members__)
                    #     else:
                    #         element = -1.0
                    #     with open(self.log_name, 'a') as afile:
                    #         afile.write(f"File Type, {element}\n")
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Vendor" in file:
                    #     if file["Vendor"] != -1:
                    #         element = file["Vendor"].value / len(Vendor.__members__)
                    #     else:
                    #         element = -1.0
                    #     with open(self.log_name, 'a') as afile:
                    #         afile.write(f"File Vendor, {element}\n")
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Version" in file:
                    #     if file["Version"] != -1:
                    #         element = file["Version"].value / len(FileVersion.__members__)
                    #     else:
                    #         element = -1.0
                    #     with open(self.log_name, 'a') as afile:
                    #         afile.write(f"File Version, {element}\n")
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Username" in file:
                    #     element = file["Username"]
                    #     with open(self.log_name, 'a') as afile:
                    #         afile.write(f"File Username, {element}\n")
                    #     if element not in self.username:
                    #         self.username[element] = len(self.username)
                    #     element = self.username[element]
                    #     flat_obs.append(float(element))
                    # else:
                        # flat_obs.append(-1.0)

                    # if "Group Name" in file:
                    #     element = file["Group Name"]
                    #     with open(self.log_name, 'a') as afile:
                    #         afile.write(f"File Group Name, {element}\n")
                    #     if element not in self.group_name:
                    #         self.group_name[element] = len(self.group_name)
                    #     element = self.group_name[element]
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Last Modified Time" in file:
                    #     #TODO work out how to normalise this value
                    #     element = -1 #(file["Last Modified Time"] - datetime(2020, 1, 1)).total_seconds()
                        
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "User Permissions" in file:
                    #     element = file["User Permissions"]
                    #     with open(self.log_name, 'a') as afile:
                    #         afile.write(f"File User Permissions, {element}\n")
                    #     flat_obs.append(float(element)/7)
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Group Permissions" in file:
                    #     element = file["Group Permissions"]
                    #     with open(self.log_name, 'a') as afile:
                    #         afile.write(f"File Group Permissions, {element}\n")
                    #     flat_obs.append(float(element)/7)
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Default Permissions" in file:
                    #     element = file["Default Permissions"]
                    #     with open(self.log_name, 'a') as afile:
                    #         afile.write(f"File Default Permissions, {element}\n")
                    #     flat_obs.append(float(element)/7)
                    # else:
                    #     flat_obs.append(-1.0)

                if "User Info" not in host:
                    host["User Info"] = []
                while len(host["User Info"]) < self.MAX_USERS:
                    host["User Info"].append({})

                if len(host["User Info"]) > self.MAX_USERS:
                    with open(self.log_name, 'a') as file: 
                        file.write(f'MAX_USERS: {len(host["User Info"])}\n')
                while len(host["User Info"]) > self.MAX_USERS:
                    host["User Info"].pop()
                    # raise ValueError("Too many users in observation for fixed size")

                for user_idx, user in enumerate(host['User Info']):
                    if "Username" in user:
                        element = user["Username"]
                        with open(self.log_name, 'a') as file:
                            file.write(f"User Info Username, {element}\n")
                        if element not in self.username:
                            self.username[element] = len(self.username)
                        element = self.username[element]
                        flat_obs.append(float(element))
                    else:
                        flat_obs.append(-1.0)
                    print('user info username', len(flat_obs)-1)

                    if "Password" in user:
                        element = user["Password"]
                        with open(self.log_name, 'a') as file:
                            file.write(f"User Info Password, {element}\n")
                        if element not in self.password:
                            self.password[element] = len(self.password)
                        element = self.password[element]
                        flat_obs.append(float(element))
                    else:
                        flat_obs.append(-1.0)
                    print('user info Password', len(flat_obs)-1)
                    # if "Password Hash" in user:
                    #     element = user["Password Hash"]
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"User Info Password Hash, {element}\n")
                    #     if element not in self.password_hash:
                    #         self.password_hash[element] = len(self.password_hash)
                    #     element = self.password_hash[element]
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Password Hash Type" in user:
                    #     if user["Password Hash Type"] != -1:
                    #         element = user["Password Hash Type"].value / len(PasswordHashType.__members__)
                    #     else:
                    #         element = -1.0
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"User Info Password Hash Type, {element}\n")
                    #     flat_obs.append(float(element))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "UID" in user:
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"User Info UID, {user['UID']}\n")
                    #     flat_obs.append(float(user["UID"]))
                    # else:
                    #     flat_obs.append(-1.0)

                    # if "Logged in" in user:
                    #     with open(self.log_name, 'a') as file:
                    #         file.write(f"User Info Logged in, {user['Logged in']}\n")
                    #     flat_obs.append(float(user["Logged in"]))
                    # else:
                    #     flat_obs.append(-1.0)

                    if "Groups" not in user:
                        user["Groups"] = []
                    while len(user["Groups"]) < self.MAX_GROUPS:
                        user["Groups"].append({})
                    
                    if len(user['Groups']) > self.MAX_GROUPS:
                        with open(self.log_name, 'a') as file: 
                            file.write(f'MAX_GROUPS: {len(user["Groups"])}\n')
                    while len(user['Groups']) > self.MAX_GROUPS:
                        user['Groups'].pop()
                        # raise ValueError("Too many groups in observation for fixed size")
                    for group_idx, group in enumerate(user["Groups"]):
                        # if 'Builtin Group' in group:
                        #     if group["Builtin Group"] != -1:  # TODO test if this is ever not true
                        #         element = group["Builtin Group"].value / len(BuiltInGroups.__members__)
                        #     else:
                        #         element = -1.0
                        #     with open(self.log_name, 'a') as file:
                        #         file.write(f"Groups Builtin Group, {element}\n")
                        #     flat_obs.append(float(element))
                        # else:
                        #     flat_obs.append(-1.0)

                        # if "Group Name" in group:
                        #     element = user["Group Name"]
                        #     with open(self.log_name, 'a') as file:
                        #         file.write(f"Groups Group Name, {element}\n")
                        #     if element not in self.group_name:
                        #         self.group_name[element] = len(self.group_name)
                        #     element = self.group_name[element]
                        #     flat_obs.append(float(element))
                        # else:
                        #     flat_obs.append(-1.0)

                        if "GID" in group:
                            with open(self.log_name, 'a') as file:
                                file.write(f"Groups GID, {group['GID']}\n")
                            flat_obs.append(float(group["GID"]))
                        else:
                            flat_obs.append(-1.0)

                if "Sessions" not in host:
                    host["Sessions"] = []
                while len(host["Sessions"]) < self.MAX_SESSIONS:
                    host["Sessions"].append({})

                if len(host["Sessions"]) > self.MAX_SESSIONS and len(obs.keys())!=14:
                    with open(self.log_name, 'a') as file: 
                            file.write(f'MAX_SESSIONS: {len(host["Sessions"])}\n')
                while len(host["Sessions"]) > self.MAX_SESSIONS:
                    host["Sessions"].pop()
                    # raise ValueError("Too many sessions in observation for fixed size")

                for session_idx, session in enumerate(host['Sessions']):
                    if "Username" in session:
                        element = session["Username"]
                        with open(self.log_name, 'a') as file:
                                file.write(f"Sessions Username, {element}\n")
                        if element not in self.username:
                            self.username[element] = len(self.username)
                        element = self.username[element]
                        flat_obs.append(float(element))
                    else:
                        flat_obs.append(-1.0)

                    if "Type" in session:
                        if session["Type"] != -1:
                            element = session["Type"].value/len(SessionType.__members__)
                        else:
                            element = -1.0
                        with open(self.log_name, 'a') as file:
                                file.write(f"Sessions Type, {element}\n")
                        flat_obs.append(float(element))
                    else:
                        flat_obs.append(-1.0)

                    if "ID" in session:
                        with open(self.log_name, 'a') as file:
                                file.write(f"Sessions ID, {session['ID']}\n")
                        flat_obs.append(float(session["ID"])/20)
                    else:
                        flat_obs.append(-1.0)

                    if "Timeout" in session:
                        with open(self.log_name, 'a') as file:
                                file.write(f"Sessions Timeout, {session['Timeout']}\n")
                        flat_obs.append(float(session["Timeout"]))
                    else:
                         flat_obs.append(-1.0)

                    if "PID" in session:
                        element = session["PID"]
                        with open(self.log_name, 'a') as file:
                                file.write(f"Sessions PID, {element}\n")
                        if element not in self.session_pid:
                            self.session_pid[element] = len(self.session_pid)
                            # with open(self.log_name, 'a') as file:
                            #     file.write(f"session_pid, {len(self.session_pid)}\n")
                        element = self.session_pid[element]
                        flat_obs.append(float(element))
                        # flat_obs.append(float(session["PID"])/32768)
                    else:
                        flat_obs.append(-1.0)

                if 'Interface' not in host:
                    host["Interface"] = []
                while len(host["Interface"]) < self.MAX_INTERFACES:
                    host["Interface"].append({})
                
                if len(host["Interface"]) > self.MAX_INTERFACES:
                    with open(self.log_name, 'a') as file: 
                            file.write(f'MAX_INTERFACES: {len(host["Interface"])}\n')
                while len(host["Interface"]) > self.MAX_INTERFACES:
                    host["Interface"].pop()
                    # raise ValueError("Too many interfaces in observation for fixed size")

                if 'Interface' in host:
                    for interface_idx, interface in enumerate(host['Interface']):
                        # if "Interface Name" in interface:
                        #     element = interface["Interface Name"]
                        #     if element not in self.interface_name:
                        #         self.interface_name[element] = len(self.interface_name)
                        #     element = self.interface_name[element]
                        #     flat_obs.append(float(
                        #             element))
                        # else:
                        #      flat_obs.append(-1.0)

                        # if "Subnet" in interface:
                        #     element = interface["Subnet"].network_address
                        #     if element not in self.subnet_netadd:
                        #         self.subnet_netadd[element] = len(self.subnet_netadd)
                        #         # with open(self.log_name, 'a') as file:
                        #         #     file.write(f"subnet_netadd, {len(self.subnet_netadd)}\n")
                        #     element = self.subnet_netadd[element]
                        #     flat_obs.append(float(element))
                        #     # flat_obs.append(float(int(element.network_address))/4294967296)
                        #     element = interface["Subnet"].prefixlen
                        #     if element not in self.subnet_prefix:
                        #         self.subnet_prefix[element] = len(self.subnet_prefix)
                        #         # with open(self.log_name, 'a') as file:
                        #         #     file.write(f"subnet_prefix, {len(self.subnet_prefix)}\n")
                        #     element = self.subnet_prefix[element]
                        #     flat_obs.append(float(element))
                        #     # flat_obs.append(float(int(element.prefixlen))/4294967296)
                        # else:
                        #      flat_obs.append(-1.0)
                        #      flat_obs.append(-1.0)

                        if "IP Address" in interface:
                            element = int(interface["IP Address"])
                            if element not in self.ipadd:
                                self.ipadd[element] = len(self.ipadd)
                                # with open(self.log_name, 'a') as file:
                                #     file.write(f"ipadd, {len(self.ipadd)}\n")
                            element = self.ipadd[element]
                            flat_obs.append(float(element))
                            # flat_obs.append(float(element)/4294967296)
                        else:
                             flat_obs.append(-1.0)
                        print('ip add', len(flat_obs)-1)

        # print('flat_obs', np.shape(flat_obs), flush=True)
        
        ## change here for PCA
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # # print('dir_path', dir_path, flush=True)
        # pca_obs = pickle.load(open(dir_path+'/pca_10000.pkl', 'rb'))
        # obs = np.reshape(flat_obs, (-1, np.shape(flat_obs)[0]))
        # obs_transformed = pca_obs.transform(obs)
        # # print('obs_transformed.squeeze()', np.shape(obs_transformed.squeeze()), flush=True)
        # return obs_transformed.squeeze()
        if self.count%2==0:
            with open('flat_obs.csv', 'a') as rb_csv:
                writer_rb_csv = writer(rb_csv)
                writer_rb_csv.writerow(flat_obs)
                rb_csv.close()
        return flat_obs

    def get_attr(self,attribute:str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        # print('in get_observation()', flush=True)
        obs = self.get_attr('get_observation')(agent)
        # new_obs = self.observation_change(obs)
        # print('new_obs', new_obs, flush=True)
        return self.observation_change(obs)

# 
# Copyright (C) 2019 EEMBC(R). All Rights Reserved
# 
# All EEMBC Benchmark Software are products of EEMBC and are provided under the
# terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
# are proprietary intellectual properties of EEMBC and its Members and is
# protected under all applicable laws, including all applicable copyright laws.  
# 
# If you received this EEMBC Benchmark Software without having a currently
# effective EEMBC Benchmark License Agreement, you must discontinue use.
# 

import platform
import subprocess
import re
import multiprocessing as mp
import sys
from eelib import constants

class SysInfo:

	def __getTotalRamInBytes(self):
		p = platform.system()
		if p == 'Linux':
			meminfo = open('/proc/meminfo').read()
			m = re.search(r'MemTotal:\s*(\d+)', meminfo)
			if m:
				return int(m.groups()[0]) * 1024
			return 'unknown'	
		elif p == 'Windows':
			cmd = ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory']
			m = subprocess.check_output(cmd).strip().split('\n')[1]
			return int(m)
		elif p == 'Darwin':
			command = 'sysctl -n hw.memsize'
			return int((subprocess.check_output(command, shell=True).strip()).decode('utf_8'))
		else:
			log.warning('Could not find total RAM on this platform: %s' % p)
		return 'unknown'

	def __getCpuName(self):
		p = platform.system()
		if p == 'Linux':
			name = ''
			command = 'lscpu'
			r = (subprocess.check_output(command, shell=True).strip()).decode('utf_8')
			m = re.search(r'Vendor ID:\s*(.+)\s*', r)
			if m:
				name = m.groups()[0]
			m = re.search(r'Model name:\s*(.+)\s*', r)
			if m:
				name = '%s %s' % (name, m.groups()[0])
			if name == '':
				return 'unkown'
			return name
		elif p == 'Windows':
			cmd = ['wmic','cpu','get', 'name']
			name = subprocess.check_output(cmd).strip().split('\n')[1]
			return name.decode('utf_8')
		elif p == 'Darwin':
			command = 'sysctl -n machdep.cpu.brand_string'
			name = subprocess.check_output(command, shell=True).strip()
			return name.decode('utf_8')
		else:
			log.warning('Could not find CPU name on this platform: %s' % p)
		return 'unknown'

	def get(self):
		# Return an ordered list instead of a dict so that printing is ordered
		return [
			[ "MLMark Version", constants.VERSION ],
			[ "Python Version", "%d.%d" % (sys.version_info[0], sys.version_info[1]) ],
			[ "CPU Name", self.__getCpuName() ],
			[ "Total Memory (MiB)", int(self.__getTotalRamInBytes() / (1024 * 1024)) ],
			[ "# of Logical CPUs", mp.cpu_count() ],
			[ "Instruction Set", platform.machine() ],
			[ "OS Platform", platform.platform() ]
		]

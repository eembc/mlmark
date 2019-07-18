# COCO annotations are very large, so use an object to keep them in memory.
#
# This code was pruned significantly because we don't need the _mask libc
# dependencies, just the index creation.
#
# This is based on the code by Tsung-Yi Lin at this GitHub repo:
# https://github.com/cocodataset/cocoapi and is under this license:
#
#--
#
# Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies, 
# either expressed or implied, of the FreeBSD Project.

import json
from collections import defaultdict
import itertools

def _isArrayLike(obj):
	return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class COCO:
	def __init__(self, annotation_file=None):
		"""
		Constructor of Microsoft COCO helper class for reading and visualizing annotations.
		:param annotation_file (str): location of annotation file
		:param image_folder (str): location to the folder that hosts images.
		:return:
		"""
		# load dataset
		self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
		self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
		if not annotation_file == None:
			dataset = json.load(open(annotation_file, 'r'))
			assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
			self.dataset = dataset
			self.createIndex()

	def createIndex(self):
		# create index
		anns, cats, imgs = {}, {}, {}
		imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
		if 'annotations' in self.dataset:
			for ann in self.dataset['annotations']:
				imgToAnns[ann['image_id']].append(ann)
				anns[ann['id']] = ann

		if 'images' in self.dataset:
			for img in self.dataset['images']:
				imgs[img['id']] = img

		if 'categories' in self.dataset:
			for cat in self.dataset['categories']:
				cats[cat['id']] = cat

		if 'annotations' in self.dataset and 'categories' in self.dataset:
			for ann in self.dataset['annotations']:
				catToImgs[ann['category_id']].append(ann['image_id'])

		# create class members
		self.anns = anns
		self.imgToAnns = imgToAnns
		self.catToImgs = catToImgs
		self.imgs = imgs
		self.cats = cats

	def loadCats(self, ids=[]):
		"""
		Load cats with the specified ids.
		:param ids (int array)       : integer ids specifying cats
		:return: cats (object array) : loaded cat objects
		"""
		if _isArrayLike(ids):
			return [self.cats[id] for id in ids]
		elif type(ids) == int:
			return [self.cats[ids]]

	def loadImgs(self, ids=[]):
		"""
		Load anns with the specified ids.
		:param ids (int array)       : integer ids specifying img
		:return: imgs (object array) : loaded img objects
		"""
		if _isArrayLike(ids):
			return [self.imgs[id] for id in ids]
		elif type(ids) == int:
			return [self.imgs[ids]]
			
	def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
		"""
		Get ann ids that satisfy given filter conditions. default skips that filter
		:param imgIds  (int array)     : get anns for given imgs
			   catIds  (int array)     : get anns for given cats
			   areaRng (float array)   : get anns for given area range (e.g. [0 inf])
			   iscrowd (boolean)       : get anns for given crowd label (False or True)
		:return: ids (int array)       : integer array of ann ids
		"""
		imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
		catIds = catIds if _isArrayLike(catIds) else [catIds]

		if len(imgIds) == len(catIds) == len(areaRng) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(imgIds) == 0:
				lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
				anns = list(itertools.chain.from_iterable(lists))
			else:
				anns = self.dataset['annotations']
			anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
			anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
		if not iscrowd == None:
			ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
		else:
			ids = [ann['id'] for ann in anns]
		return ids


	def loadAnns(self, ids=[]):
		"""
		Load anns with the specified ids.
		:param ids (int array)       : integer ids specifying anns
		:return: anns (object array) : loaded ann objects
		"""
		if _isArrayLike(ids):
			return [self.anns[id] for id in ids]
		elif type(ids) == int:
			return [self.anns[ids]]

#!/bin/bash
curl -L -o ./essentials/loftr-repo.zip\
  https://www.kaggle.com/api/v1/datasets/download/morizin/loftr-repo 

unzip ./essentials/loftr-repo.zip -d ./essentials/loftr-repo >> /dev/null
rm ./essentials/loftr-repo.zip


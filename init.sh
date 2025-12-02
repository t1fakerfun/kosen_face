#! /usr/bin/env bash

[ -f $HOME/bunruiki/facial_features_dataset.csv ] && rm $HOME/bunruiki/facial_features_dataset.csv
[ -f $HOME/bunruiki/faculty_classifier_model.pkl ] && rm $HOME/bunruiki/faculty_classifier_model.pkl
[ -d $HOME/bunruiki/falses ] && rm -rf $HOME/bunruiki/falses

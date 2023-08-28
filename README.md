# KIST

- 연구 목표 : 왼쪽 오른쪽 귀로 서로 다른 소리를 들을 때 어떤 소리에 집중하고 있는지 뇌파와 딥러닝을 통해 확인하기

Nerual Trk
  : 60channel의 EEG data와 1channel의 envelope data를 concat시켜 CNN 기반의 모델에 통과시켜 유사도를 구한 후 이중 분류


CLIP
  : 60channel의 EEG data를 100차원으로 encoding + 소리 데이터를 melspectrogram으로 변환 후 100차원으로 encoding
  코사인 유사도를 구하는 CLIP 방식을 차용하여 일반화 시도


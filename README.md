# pstage_01_image_classification

### image data 추가
- 다음과 같은 구성으로 Data 파일 추가하여 학습
data
  ㄴ train
      ㄴ images
      ㄴ train.csv
  ㄴ eval
      ㄴ images
      ㄴ info.csv


### 헙업 규칙

#### Commit Convention
<타입>: <제목>

제목은 50글자를 넘기지 않으며, 문장의 끝에 마침표를 넣지 않는다.
커밋 제목에는 과거시제를 사용하지 않는다.
예시 - feat: 로그인 기능 추가

본문은 위에 작성
본문은 선택사항. 제목 외에 전달하고 싶은 추가 정보가 있을 때 활용한다. 단, 최대 72글자를 넘지 않을 것.
------------------
    제목과 본문을 한 줄 띄워 분리하기
    본문은 "어떻게" 보다 "무엇을", "왜"를 설명한다.
    본문에 여러줄의 메시지를 작성할 땐 "-"로 구분
------------------

꼬릿말은 아래에 작성: ex) #이슈 번호
이슈 번호는 github의 issue 번호를 사용합니다.

 --- COMMIT END ---
<타입>
  feat    : 새로운 기능 추가
  fix     : 버그 수정
  refactor: 코드 리팩토링
  style   : 스타일 (세미콜론 누락이나 코드 포맷팅 등 코드 변경이 없는 경우)
  docs    : 문서 추가, 수정, 삭제
  test    : 테스트 (테스트 코드 추가, 수정, 삭제)
  chore   : 기타 변경사항 (빌드 스크립트 수정, 패키지 매니저 수정 등)
------------------


#### Issue & Commit
1. Issue에 개발 / 수정할 내용 올리기
2. 부여된 Issue번호를 이용하여 브랜치 만들기   ex) #00feature/cutmix
'''
git checkout -b #00feature/cutmix
'''
3. 코드 변경 후 commit
'''
git add .
git commit -m "커밋 로그" # commit log는 Commit Convention 참고하여 작성    ex) docs:README Commit convention 추가
git push -u origin #00feature/cutmix
'''
4. pull request 보내기 (#00feature/cutmix -> master)
5. 팀원 확인 후 팀장이 Merge
6. merge 시 발생하는 conflict 해결
7. 해당 Issue close



## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`

본 프로젝트는 VQA(Vision Question Answering)라는 딥러닝 세부분야와 관련된 프로젝트 입니다.

VQA는 multi modal problem으로서 computer vision과 natural language을 동시에 다루고 있습니다.

주어진 이미지와 관련된 질문을 컴퓨터에게 자연어 상태로 입력을 하면, 컴퓨터는 그에 대한 답변을 자연어 상태로 출력해 줍니다.

예시) <br/>

- 입력: 남자가 입고 있는 것은 무엇인가? / 여자가 쓰고 있는 모자의 색은? / 남자는 무엇을 하고 있는가?

- 출력: 코트 / 파란색 / 축구

본 프로젝트는 건국대학교 컴퓨터공학과 하영국 교수님 지도 아래 진행되었습니다. 프로젝트의 내용은 논문으로도 작성하여 2020 IEEE Bigcomp conference에 제출하였으나 아쉽게도 accept 되지 않았습니다. 작성했던 논문은 [링크][0]로 남깁니다.

본 프로젝트에 대한 전반적인 설명은 아래의 포스터를 참고해 주세요.

![KOCO VQA](/포스터.PNG)

[0]: /KOCO_VQA_Korean_VQA_with_Advanced_Counting.pdf

<!--이번 프로젝트는 쉽지 않은 프로젝트였다. 이미지와 자연어처리를 동시에 다뤄야하는 분야였기 때문이다.베이스로 삼은 코드(https://github.com/Cyanogenoid/vqa-counting/tree/master/vqa-v2)를 한국어에 맞게 수정, 보완하기 위해 수많은 시도를 하였다. 자연어처리 부분에 BERT를 적용해보기도 하고, 여러 fusion 방식(Linearsum, MLB, MCB)을 적용해보려고 노력하였다. 시도한 것 중 상당부분은 실패하였지만 결과적으로는 소기의 성과를 얻으며 프로젝트를 마칠 수 있었다.-->
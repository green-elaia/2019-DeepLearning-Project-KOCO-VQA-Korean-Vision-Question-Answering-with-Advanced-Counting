![KOCO VQA](/포스터.PNG){: width="100%" height="100%"}KOCO VQA 

<br/>

### 느낀점

이번 프로젝트는 쉽지 않은 프로젝트였다. 이미지와 자연어처리를 동시에 다뤄야하는 분야였기 때문이다.

[베이스로 삼은 코드][0]를 한국어에 맞게 수정, 보완하기 위해 수많은 시도를 하였다. 자연어처리 부분에 BERT를 적용해보기도 하고, 여러 fusion 방식(Linearsum, MLB, MCB)을 적용해보려고 노력하였다. 시도한 것 중 상당부분은 실패하였지만 결과적으로는 소기의 성과를 얻으며 프로젝트를 마칠 수 있었다.

참고. 데이터셋은 따로 업로드하지 않았다.

프로젝트 내용을 논문으로 작성해 IEEE에 제출해보기도 했지만 아쉽게도 accept이 되지는 않았다. 그래도 세계적인 저널이 요구하는 포맷에 맞춰 논문을 작성해 볼 수 있었던 좋은 기회였다. 아래는 논문을 작성하며 참고했던 논문 리스트이다.

<br/>

[1]   Antol, Stanislaw, et al. "Vqa: Visual question answering." *Proceedings of the IEEE international conference on computer vision*. 2015.

[2]   Malinowski, Mateusz, Marcus Rohrbach, and Mario Fritz. "Ask your neurons: A neural-based approach to answering questions about images." *Proceedings of the IEEE international conference on computer vision*. 2015.

[3]   Zhang, Yan, Jonathon Hare, and Adam Prügel-Bennett. "Learning to count objects in natural images for visual question answering." *arXiv preprint arXiv:1802.05766* (2018).

[4]   Kazemi, Vahid, and Ali Elqursh. "Show, ask, attend, and answer: A strong baseline for visual question answering." *arXiv preprint arXiv:1704.03162* (2017).

[5]   He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[6]   Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-zterm memory." *Neural computation* 9.8 (1997): 1735-1780.

[7]   Trott, Alexander, Caiming Xiong, and Richard Socher. "Interpretable counting for visual question answering." *arXiv preprint arXiv:1712.08697* (2017).

[8]   Yang, Zichao, et al. "Stacked attention networks for image question answering." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[9]   Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." *arXiv preprint arXiv:1409.0473* (2014).

[10]  Bae, Jangseong, and Changki Lee. "Korean VQA with Deep learning." *Annual Conference on Human and Language Technology*. Human and Language Technology, 2018.

[11]  Ren, Mengye, Ryan Kiros, and Richard Zemel. "Exploring models and data for image question answering." *Advances in neural information processing systems*. 2015.

[12]  Tu, Zhaopeng, et al. "Context gates for neural machine translation." *Transactions of the Association for Computational Linguistics* 5 (2017): 87-99.

[13]  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." *Advances in neural information processing systems*. 2012.

[14]  Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling." *arXiv preprint arXiv:1412.3555* (2014).

[15]  Lee, Hyoung-Gyu, et al. "papago: A machine translation service with word sense disambiguation and currency conversion." *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: System Demonstrations*. 2016.

[16]  Yang, Jae-Woo, and Youngjik Lee. "Toward translating Korean speech into other languages." *Proceeding of Fourth International Conference on Spoken Language Processing. ICSLP'96*. Vol. 4. IEEE, 1996.

[17]  Kudo, Taku. "Mecab: Yet another part-of-speech and morphological analyzer." *http://mecab. sourceforge. jp* (2006).

[18]  Park, Eunjeong L., and Sungzoon Cho. "KoNLPy: Korean natural language processing in Python." *Proceedings of the 26th Annual Conference on Human & Cognitive Language Technology*. Vol. 6. 2014.

[19]  Goldberg, Yoav, and Omer Levy. "word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method." *arXiv preprint arXiv:1402.3722* (2014).

 

[0]: https://github.com/Cyanogenoid/vqa-counting/tree/master/vqa-v2


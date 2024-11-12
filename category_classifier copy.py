import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import re
import os
import json
from datetime import datetime
import joblib

class CategoryClassifier:
    def __init__(self):
        # TF-IDF 벡터라이저 설정 개선
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 5),      # 1~5글자로 확장
            min_df=2,
            analyzer='char_wb',
            sublinear_tf=True,
            max_features=5000        # 주요 특성만 사용
        )
        
        # 분류기 설정 강화
        self.base_classifiers = [
            ('rf', RandomForestClassifier(
                n_estimators=300,     # 트리 수 증가
                max_depth=10,         # 적절한 깊이 제한
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )),
            ('lr', LogisticRegression(
                C=1.0,               # 규제 강도 조정
                max_iter=2000,
                class_weight='balanced',
                random_state=42
            )),
            ('svc', SVC(
                kernel='rbf',        # RBF 커널로 변경
                C=10.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            ))
        ]
        
        # 앙상블 분류기 설정
        self.classifier = VotingClassifier(
            estimators=self.base_classifiers,
            voting='soft'  # 'hard'에서 'soft'로 변경
        )
        
        # 예측 이력 저장
        self.prediction_history = []

        # 카테고리 정의와 함께 키워드 매핑
        self.category_patterns = {
            'C001001': {
                'name': '투명 케이스',
                'keywords': ['투명', '클리어', '맥세이프', '크리스탈', '글라스', '초강력'],
                'patterns': [r'투명.*케이스', r'클리어.*케이스', r'맥세이프.*클리어.*케이스', r'크리스탈.*케이스'],
                'priority_keywords': ['클리어', '투명']
            },
            'C001002': {
                'name': '젤리/실리콘 케이스',
                'keywords': ['젤리', '실리콘', '소프트', 'tpu', '슬림'],
                'patterns': [r'젤리.*케이스', r'실리콘.*케이스', r'소프트.*케이스', r'tpu.*케이스'],
                'priority_keywords': ['젤리', '실리콘']
            },
            'C001003': {
                'name': '카드 수납 케이스',
                'keywords': ['카드', '수납', '포켓', '지갑', '월렛'],
                'patterns': [r'카드.*케이스', r'수납.*케이스', r'포켓.*케이스', r'지갑.*케이스']
            },
            'C001004': {
                'name': '디자인/캐릭터 케이스',
                'keywords': ['디자인', '캐릭터', '패턴', '그림', '프린팅'],
                'patterns': [r'디자인.*케이스', r'캐릭터.*케이스', r'패턴.*케이스']
            },
            'C001005': {
                'name': '핑거톡/핑거링 케이스',
                'keywords': ['핑거톡', '핑거링', '스마트링', '스트랩'],
                'patterns': [r'핑거톡.*케이스', r'핑거링.*케이스', r'스마트링.*케이스']
            },
            'C001006': {
                'name': '다이어리 케이스',
                'keywords': ['다이어리', '수첩', '플립', '가죽'],
                'patterns': [r'다이어리.*케이스', r'가죽.*케이스']
            },
            'C001007': {
                'name': '플립 케이스',
                'keywords': ['플립', '커버'],
                'patterns': [r'플립.*케이스', r'플립.*커버']
            },
            'C001008': {
                'name': '폴드 케이스',
                'keywords': ['폴드', 'z폴드', '갤럭시폴드', '힌지'],
                'patterns': [
                    r'z.*폴드.*케이스',
                    r'갤럭시.*폴드.*케이스',
                    r'폴드.*케이스'
                ],
                'exclude_keywords': ['보조배터리', '충전기']  # 제외할 키워드
            },
            'C001009': {
                'name': '하드/범퍼 케이스',
                'keywords': ['하드', '범퍼', '강화'],
                'patterns': [r'하드.*케이스', r'범퍼.*케이스(?!.*젤리)', r'강화.*케이스(?!.*필름)'],
                'priority_keywords': ['하드', '범퍼']
            },
            'C001010': {
                'name': '아이패드/갤럭시탭 케이스',
                'keywords': ['아이패드', '갤럭시탭', '태블릿'],
                'patterns': [r'아이패드.*케이스', r'갤럭시탭.*케이스', r'태블릿.*케이스']
            },
            'C002001': {
                'name': '아이폰 액정필름',
                'keywords': ['아이폰', '강화유리', '필름', '보호필름', '아이폰필름'],
                'patterns': [r'아이폰.*필름', r'아이폰.*강화유리', r'강화유리.*필름.*아이폰'],
                'priority_keywords': ['강화유리', '필름']
            },
            'C002002': {
                'name': '갤럭시 액정필름',
                'keywords': ['갤럭시', '강화유리', '필름', '보호필름', '갤럭시필름'],
                'patterns': [r'갤럭시.*필름', r'갤럭시.*강화유리']
            },
            'C002003': {
                'name': '플립 액정필름',
                'keywords': ['플립', '강화유리', '필름', '보호필름'],
                'patterns': [r'플립.*필름', r'플립.*강화유리']
            },
            'C002004': {
                'name': '폴드 액정필름',
                'keywords': ['폴드', '강화유리', '필름', '보호필름'],
                'patterns': [r'폴드.*필름', r'폴드.*강화유리']
            },
            'C002005': {
                'name': '카메라 렌즈 보호필름',
                'keywords': ['카메라', '렌즈', '보호필름', '카메라필름'],
                'patterns': [r'카메라.*필름', r'렌즈.*필름', r'카메라.*보호']
            },
            'C002006': {
                'name': '아이패드/갤럭시탭 액정필름',
                'keywords': ['아이패드', '갤럭시탭', '태블릿', '필름'],
                'patterns': [r'아이패드.*필름', r'갤럭시탭.*필름', r'태블릿.*필름']
            },
            'C003001': {
                'name': '핑거톡',
                'keywords': ['핑거톡', '스마트톡', '그립톡'],
                'patterns': [r'.*핑거톡', r'.*스마트톡', r'.*그립톡']
            },
            'C003002': {
                'name': '핑거링',
                'keywords': ['핑거링', '스마트링', '그립링'],
                'patterns': [r'.*핑거링', r'.*스마트링', r'.*그립링']
            },
            'C003003': {
                'name': '스트랩',
                'keywords': ['스트랩', '목걸이'],
                'patterns': [r'.*스트랩', r'폰.*목걸이']
            },
            'C003004': {
                'name': '카드홀더',
                'keywords': ['카드홀더', '카드수납', '카드포켓'],
                'patterns': [r'카드.*홀더', r'카드.*포켓']
            },
            'C004001': {
                'name': '가정용 충전',
                'keywords': ['가정용', '충전기', '어댑터', '전원어댑터'],
                'patterns': [r'가정용.*충전기', r'충전.*어댑터']
            },
            'C004002': {
                'name': '차량용 충전기',
                'keywords': ['차량용', '시거잭', '차량충전'],
                'patterns': [r'차량용.*충전기', r'시거잭']
            },
            'C004003': {
                'name': '무선 충전기',
                'keywords': ['무선충전', '무선패드', '맥세이프'],
                'patterns': [r'무선.*충전기', r'맥세이프.*충전']
            },
            'C004004': {
                'name': '충전기 케이블',
                'keywords': ['케이블', '충전선', 'c타입', '8핀'],
                'patterns': [r'충전.*케이블', r'충전선', r'타입c']
            },
            'C004005': {
                'name': '블루투스',
                'keywords': ['블루투스', '무선연결'],
                'patterns': [r'블루투스']
            },
            'C004006': {
                'name': '이어폰',
                'keywords': ['이어폰', '무선이어폰', '블루투스이어폰'],
                'patterns': [r'.*이어폰', r'에어팟']
            },
            'C004007': {
                'name': '스피커',
                'keywords': ['스피커', '블루투스스피커'],
                'patterns': [r'.*스피커']
            },
            'C004008': {
                'name': '보조배터리',
                'keywords': ['보조배터리', '배터리팩', 'mah'],
                'patterns': [
                    r'보조배터리',
                    r'\d+\s*mah',
                    r'배터리팩'
                ],
                'exclude_keywords': ['케이스']  # 제외할 키워드
            },
            'C005001': {
                'name': '기타 거치대',
                'keywords': ['거치대', '스탠드', '받침대'],
                'patterns': [r'거치대', r'스탠드', r'받침대']
            },
            'C005002': {
                'name': '차량용 거치대',
                'keywords': ['차량용', '거치대', '차량거치'],
                'patterns': [r'차량용.*거치대', r'차량.*거치']
            },
            'C005003': {
                'name': '셀카봉/삼각대',
                'keywords': ['셀카봉', '삼각대', '셀피스틱'],
                'patterns': [r'셀카봉', r'삼각대', r'셀피스틱']
            },
            'C005004': {
                'name': '선풍기',
                'keywords': ['선풍기', '미니팬', '휴대용선풍기'],
                'patterns': [r'선풍기', r'미니팬']
            }
        }

    def normalize_text(self, text):
        """텍스트 정규화 - 더 세밀한 처리"""
        # 기본 정규화
        text = text.lower().strip()
        
        # 특수문자 처리
        text = re.sub(r'[\(\)\[\]\{\}]', ' ', text)  # 괄호류
        text = re.sub(r'[.,!?~\-_+*/=]', ' ', text)  # 문장부호
        text = re.sub(r'[^\w\s가-힣]', '', text)    # 나머지 특수문자
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 숫자 앞뒤 공백 추가
        text = re.sub(r'(\d+)', r' \1 ', text)
        
        return text.strip()

    def prepare_training_data(self):
        """학습 데이터 생성 개선"""
        texts = []
        labels = []
        
        for category_id, info in self.category_patterns.items():
            # 1. 기본 키워드 변형
            for keyword in info['keywords']:
                variations = [
                    keyword,
                    f"{keyword} {info['name']}",
                    f"{info['name']} {keyword}",
                    # 실제 상품명 패턴 추가
                    f"프리미엄 {keyword} 케이스",
                    f"정품 {keyword} 케이스",
                    f"고급 {keyword}",
                    f"{keyword} 신상품",
                    # 모델명 조합
                    f"{keyword} 갤럭시 s24",
                    f"{keyword} 아이폰15",
                    f"{keyword} 갤럭시 z폴드5",
                    # 브랜드명 조합
                    f"베온 {keyword}",
                    f"맥씨 {keyword}",
                    f"내셔널지오그래픽 {keyword}",
                    # 상품 특성 조합
                    f"강화 {keyword}",
                    f"슬림 {keyword}",
                    f"하이브리드 {keyword}"
                ]
                
                texts.extend(variations)
                labels.extend([category_id] * len(variations))
            
            # 2. 패턴 기반 데이터 생성
            for pattern in info['patterns']:
                base_pattern = pattern.replace('.*', ' ')
                variations = [
                    base_pattern,
                    f"신상 {base_pattern}",
                    f"프리미엄 {base_pattern}",
                    f"{base_pattern} 신제품",
                    # 실제 상품명 패턴
                    f"{base_pattern} SM-S921",
                    f"{base_pattern} iPhone 15 Pro",
                    f"{base_pattern} SM-F946",
                    # 브랜드 + 특성 조합
                    f"베온 슬림핏 {base_pattern}",
                    f"맥씨 하이브리드 {base_pattern}",
                    f"프리모 강화 {base_pattern}"
                ]
                
                texts.extend(variations)
                labels.extend([category_id] * len(variations))
            
            # 3. 카테고리별 특수 패턴 추가
            if category_id == 'C002001':  # 아이폰 액정필름
                special_patterns = [
                    "강화유리 필름 아이폰15",
                    "아이폰 프로 강화유리",
                    "아이폰 액정보호 필름",
                    "맥씨 아이폰 강화유리"
                ]
                texts.extend(special_patterns)
                labels.extend([category_id] * len(special_patterns))
            
            elif category_id == 'C001008':  # 폴드 케이스
                special_patterns = [
                    "갤럭시 z폴드5 케이스",
                    "폴드5 힌지 케이스",
                    "폴드 스탠드 케이스",
                    "z폴드 마그네틱 케이스"
                ]
                texts.extend(special_patterns)
                labels.extend([category_id] * len(special_patterns))
        
        # 4. 데이터 증강 (노이즈 추가)
        augmented_texts = []
        augmented_labels = []
        for text, label in zip(texts, labels):
            # 띄어쓰기 변형
            augmented_texts.append(text.replace(' ', ''))
            augmented_labels.append(label)
            # 오타 추가
            if '케이스' in text:
                augmented_texts.append(text.replace('케이스', '케이스'))
                augmented_labels.append(label)
        
        texts.extend(augmented_texts)
        labels.extend(augmented_labels)
        
        return texts, labels

    def train(self, texts=None, labels=None, validate=True):
        """모델 학습 - 검증 기능 포함"""
        if texts is None or labels is None:
            texts, labels = self.prepare_training_data()
        
        # 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 텍스트 정규화
        X_train_processed = [self.normalize_text(text) for text in X_train]
        X_val_processed = [self.normalize_text(text) for text in X_val]
        
        # 특징 추출
        X_train_features = self.vectorizer.fit_transform(X_train_processed)
        X_val_features = self.vectorizer.transform(X_val_processed)
        
        # 모델 학습
        self.classifier.fit(X_train_features, y_train)
        
        if validate:
            # 검증
            val_predictions = self.classifier.predict(X_val_features)
            accuracy = sum(y_pred == y_true for y_pred, y_true in zip(val_predictions, y_val)) / len(y_val)
            print(f"\n검증 정확도: {accuracy:.4f}")
            
            # 상세 성능 보고서
            print("\n분류 보고서:")
            print(classification_report(y_val, val_predictions))
            
            # 교차 검증
            all_features = self.vectorizer.transform([self.normalize_text(text) for text in texts])
            cv_scores = cross_val_score(self.classifier, all_features, labels, cv=5)
            print(f"\n교차 검증 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    def predict(self, texts):
        """여러 텍스트에 대한 예측 처리"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        self.prediction_history = []  # 예측 이력 초기화
        
        for text in texts:
            text_lower = text.lower()
            
            # 각 카테고리별로 점수 계산
            scores = {}
            for category_id, category in self.category_patterns.items():
                # 제외 키워드 체크
                if any(keyword in text_lower for keyword in category.get('exclude_keywords', [])):
                    continue
                    
                # 키워드 매칭
                keyword_matches = sum(1 for keyword in category['keywords'] if keyword.lower() in text_lower)
                
                # 패턴 매칭
                pattern_matches = sum(1 for pattern in category['patterns'] if re.search(pattern, text_lower))
                
                # 총점 계산
                total_score = keyword_matches + pattern_matches
                if total_score > 0:
                    scores[category_id] = total_score

            # 가장 높은 점수의 카테고리 선택
            if scores:
                best_category = max(scores.items(), key=lambda x: x[1])
                category_id = best_category[0]
                confidence = min(0.5 + (best_category[1] * 0.1), 1.0)
            else:
                # 기본값 설정
                category_id = list(self.category_patterns.keys())[0]
                confidence = 0.1

            result = {
                'text': text,
                'category_id': category_id,
                'category_name': self.category_patterns[category_id]['name'],
                'confidence': confidence
            }
            
            # 결과를 리스트와 예측 이력에 추가
            results.append(result)
            self.prediction_history.append(result)
        
        return results

    def _log_prediction(self, result):
        """예측 결과 로깅"""
        self.prediction_history.append({
            'text': result['text'],
            'category_id': result['category_id'],
            'category_name': result['category_name'],
            'confidence': result['confidence'],
            'timestamp': datetime.now()
        })

    def save_model(self, directory="category_classifier"):
        """모델 저장"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(directory, f"model_{timestamp}")
        os.makedirs(model_path)
        
        # 모델 저장
        joblib.dump(self.vectorizer, os.path.join(model_path, "vectorizer.pkl"))
        joblib.dump(self.classifier, os.path.join(model_path, "classifier.pkl"))
        
        # 설정 및 이력 저장
        config = {
            'category_patterns': self.category_patterns,
            'prediction_history': [
                {
                    'text': p['text'],
                    'category_id': p['category_id'],
                    'category_name': p['category_name'],
                    'confidence': p['confidence'],
                    'timestamp': p['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                }
                for p in self.prediction_history
            ]
        }
        
        with open(os.path.join(model_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"모델이 저장되었습니다: {model_path}")
        return model_path

    @classmethod
    def load_model(cls, model_path):
        """모델 로드"""
        instance = cls()
        
        # 모델 로드
        instance.vectorizer = joblib.load(os.path.join(model_path, "vectorizer.pkl"))
        instance.classifier = joblib.load(os.path.join(model_path, "classifier.pkl"))
        
        # 설정 로드
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
            instance.category_patterns = config['category_patterns']
            instance.prediction_history = [
                {
                    'text': p['text'],
                    'category_id': p['category_id'],
                    'category_name': p['category_name'],
                    'confidence': p['confidence'],
                    'timestamp': datetime.strptime(p['timestamp'], "%Y-%m-%d %H:%M:%S")
                }
                for p in config['prediction_history']
            ]
        
        return instance

def main():
    # 테스트 데이터
    test_texts = [
        "맥씨 강화유리 필름 (5매) / iPhone 16 Plus / 아이폰 16 플러스",
        "크레이지 범퍼 케이스 / SM-S721 / 갤럭시 S24 FE",
        "DC 투카드 맥세이프 범퍼 케이스 / SM-S928 / 갤럭시 S24 울트라",
        "프리모 방탄톡 젤리 케이스 / SM-S721 / 갤럭시 S24 FE",
        "내셔널지오그래픽 마그네틱 힌지 커버 슬림 스탠드 Z폴드 케이스 (with S펜 홀더) / SM-F956 / 갤럭시 Z폴드6",
        "187. 내셔널지오그래픽 마그네틱 힌지 커버 슬림 스탠드 Z폴드 케이스 (with S펜 홀더) / SM-F946 / 갤럭시 Z폴드5",
        "299. 감성 다이어리 케이스 / SM-A336 / 갤럭시 A33 (5G)",
        "303. 세븐 포켓 다이어리 케이스 / SM-A155N / 갤럭시 A15 LTE",
        "003. 베온 초강력 맥세이프 클리어 케이스 iPhone 16 Plus 아이폰 16 플러스",
        "420. 베온 더셀 접이식 포트 도킹형 보조배터리 4800mAh (메인 C타입 젠더+8핀 케이블)"
    ]

    # 모델 파일이 저장된 경로 설정
    MODEL_DIR = "category_classifier"
    
    try:
        # 가장 최근의 모델 폴더 찾기
        if os.path.exists(MODEL_DIR):
            model_folders = [
                os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR)
                if d.startswith("model_")
            ]
            if model_folders:
                latest_model = max(model_folders, key=os.path.getctime)
                print(f"저장된 모델을 불러오는 중... ({latest_model})")
                classifier = CategoryClassifier.load_model(latest_model)
                print("모델 로드 완료!")
            else:
                raise FileNotFoundError("모델 폴더가 비어있습니다.")
        else:
            raise FileNotFoundError("모델 디렉토리가 없습니다.")
            
    except FileNotFoundError as e:
        print(f"저장된 모델을 찾을 수 없습니다. 새로운 모델을 학습합니다... (사유: {str(e)})")
        classifier = CategoryClassifier()
        classifier.train()
        
        # 학습된 모델 저장
        model_path = classifier.save_model(MODEL_DIR)
        print(f"새로운 모델이 저장되었습니다: {model_path}")

    # 예측 실행
    print("\n상품 분류 시작...")
    results = classifier.predict(test_texts)
    
    # 결과 출력
    print("\n분류 결과:")
    for result in results:
        print("\n입력:", result['text'])
        print(f"카테고리: [{result['category_id']}] {result['category_name']}")
        print(f"신뢰도: {result['confidence']:.4f}")

    # 성능 분석 (옵션) - 예외 처리 추가
    print("\n\n전체 예측 통계:")
    total_predictions = len(classifier.prediction_history)
    if total_predictions > 0:
        unique_categories = len(set(p['category_id'] for p in classifier.prediction_history))
        avg_confidence = sum(p['confidence'] for p in classifier.prediction_history) / total_predictions
        
        print(f"총 예측 건수: {total_predictions}")
        print(f"분류된 카테고리 수: {unique_categories}")
        print(f"평균 신뢰도: {avg_confidence:.4f}")
    else:
        print("예측 결과가 없습니다.")

if __name__ == "__main__":
    main()
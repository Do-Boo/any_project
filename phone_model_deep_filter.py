import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import re
import os
import json
from datetime import datetime

class PhoneModelClassifier:
    def __init__(self):
        # TF-IDF 벡터라이저 설정
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),      # 1~4글자 단위로 토큰화
            min_df=2,                # 최소 2번 이상 등장하는 단어만 사용
            analyzer='char_wb',       # 문자 단위 분석
            sublinear_tf=True        # tf 값에 로그 적용
        )
        
        # 기본 분류기 설정
        self.base_classifiers = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            )),
            ('lr', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )),
            ('svc', LinearSVC(
                class_weight='balanced',
                random_state=42,
                dual=False
            ))
        ]
        
        # 앙상블 분류기 설정
        self.classifier = VotingClassifier(
            estimators=self.base_classifiers,
            voting='hard'  # 다수결 투표 방식
        )
        
        # 예측 이력 저장
        self.prediction_history = []
        
        # 모델 패턴 정의
        self.model_patterns = {
            'iphone': r'(?:아이폰|iphone)\s*(\d+)(?:\s*(pro\s*max|max|pro|plus|\s)+)*',
            'galaxy_s': r'(?:갤럭시|galaxy)\s*s\s*(\d+)(?:\s*(ultra|plus|\s)+)*',
            'galaxy_a': r'(?:갤럭시|galaxy)\s*a\s*(\d+)(?:\s*(fe|에프이))*',
            'galaxy_z_flip': r'(?:갤럭시|galaxy)\s*z\s*(?:flip|플립)\s*(\d+)',
            'galaxy_z_fold': r'(?:갤럭시|galaxy)\s*z\s*(?:fold|폴드)\s*(\d+)'
        }
        
        # 변형 패턴 정의
        self.variations = {
            'iphone': {
                'prefix': ['아이폰', 'iPhone', '아이폰 ', 'iPhone ', '아이폰', 'iphone'],
                'space': ['', ' '],
                'variants': {
                    'pro max': ['프로맥스', '프로 맥스', 'pro max', 'promax', 'pro max', '프로 맥스'],
                    'pro': ['프로', 'pro'],
                    'plus': ['플러스', 'plus'],
                    'max': ['맥스', 'max']
                }
            },
            'galaxy_s': {
                'prefix': ['갤럭시', 'Galaxy', '갤럭시 ', 'Galaxy ', '갤럭시', 'galaxy'],
                'space': ['', ' '],
                'variants': {
                    'ultra': ['울트라', 'ultra'],
                    'plus': ['플러스', 'plus'],
                    'fe': ['fe', 'FE', '에프이']
                }
            },
            'galaxy_a': {
                'prefix': ['갤럭시', 'Galaxy', '갤럭시 ', 'Galaxy ', '갤럭시', 'galaxy'],
                'space': ['', ' '],
                'variants': {
                    'fe': ['fe', 'FE', '에프이']
                }
            },
            'galaxy_z': {
                'prefix': ['갤럭시', 'Galaxy', '갤럭시 ', 'Galaxy ', '갤럭시', 'galaxy'],
                'space': ['', ' '],
                'variants': {
                    'flip': ['플립', 'flip'],
                    'fold': ['폴드', 'fold']
                }
            }
        }
    
    def normalize_model_name(self, text):
        """텍스트 정규화"""
        # 시리즈 문자(s, a, z)를 대문자로 변환
        text = re.sub(r'(?:s|에스)(\d)', r'S \1', text)  # S24 -> S 24
        text = re.sub(r'(?:a|에이)(\d)', r'A \1', text)  # A54 -> A 54
        text = re.sub(r'(?:z|제트)', r'Z', text)         # z -> Z
        
        # 기본 정규화
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # 연속된 공백을 하나로
        
        # 브랜드명 정규화 (대소문자 유지)
        text = re.sub(r'아이폰|iphone', 'iPhone', text, flags=re.IGNORECASE)
        text = re.sub(r'갤럭시|galaxy', 'Galaxy', text, flags=re.IGNORECASE)
        
        # 변형 정규화
        text = re.sub(r'프로\s*맥스|pro\s*max|promax', 'Pro Max', text, flags=re.IGNORECASE)
        text = re.sub(r'프로|pro', 'Pro', text, flags=re.IGNORECASE)
        text = re.sub(r'울트라|ultra', 'Ultra', text, flags=re.IGNORECASE)
        text = re.sub(r'플러스|plus', 'Plus', text, flags=re.IGNORECASE)
        text = re.sub(r'맥스|max', 'Max', text, flags=re.IGNORECASE)
        text = re.sub(r'플립|flip', 'Flip', text, flags=re.IGNORECASE)
        text = re.sub(r'폴드|fold', 'Fold', text, flags=re.IGNORECASE)
        text = re.sub(r'에프이|fe', 'FE', text, flags=re.IGNORECASE)
        
        # 제품 유형 정규화
        text = re.sub(r'케이스용|용케이스', '케이스', text)
        text = re.sub(r'강화유리|강화필름|보호필름', '필름', text)
        
        return text

    def extract_model_info(self, text):
        """모델 정보 추출"""
        text = self.normalize_model_name(text)
        
        # iPhone 패턴 확인
        iphone_pattern = r'(?:아이폰|iphone)\s*(\d+)\s*((?:pro\s*max|프로\s*맥스|max|맥스|pro|프로|plus|플러스)*)(?:\s*케이스|\s*강화유리|\s*필름)*'
        iphone_match = re.search(iphone_pattern, text, re.IGNORECASE)
        if iphone_match:
            number = iphone_match.group(1)
            variants = iphone_match.group(2).lower() if iphone_match.group(2) else ''
            
            if re.search(r'pro\s*max|프로\s*맥스', variants):
                return f"iphone {number} pro max"
            elif re.search(r'pro|프로', variants):
                return f"iphone {number} pro"
            elif re.search(r'plus|플러스', variants):
                return f"iphone {number} plus"
            elif re.search(r'max|맥스', variants):
                return f"iphone {number} max"
            return f"iphone {number}"
        
        # Galaxy S 패턴 확인
        galaxy_s_pattern = r'(?:갤럭시|galaxy)\s*(?:S|s|에스)?\s*(\d+)\s*((?:ultra|울트라|plus|플러스|fe|에프이)*)(?:\s*케이스|\s*강화유리|\s*필름)*'
        galaxy_s_match = re.search(galaxy_s_pattern, text, re.IGNORECASE)
        if galaxy_s_match:
            number = galaxy_s_match.group(1)
            variants = galaxy_s_match.group(2).lower() if galaxy_s_match.group(2) else ''
            
            if re.search(r'ultra|울트라', variants):
                return f"galaxy S{number} ultra"
            elif re.search(r'plus|플러스', variants):
                return f"galaxy S{number} plus"
            elif re.search(r'fe|에프이', variants):
                return f"galaxy S{number} fe"
            return f"galaxy S{number}"
        
        # Galaxy A 패턴 확인
        galaxy_a_pattern = r'(?:갤럭시|galaxy)\s*(?:A|a|에이)?\s*(\d+)\s*((?:fe|에프이)*)(?:\s*케이스|\s*강화유리|\s*필름)*'
        galaxy_a_match = re.search(galaxy_a_pattern, text, re.IGNORECASE)
        if galaxy_a_match:
            number = galaxy_a_match.group(1)
            variants = galaxy_a_match.group(2).lower() if galaxy_a_match.group(2) else ''
            
            if re.search(r'fe|에프이', variants):
                return f"galaxy A{number} fe"
            return f"galaxy A{number}"
        
        # Galaxy Z Flip/Fold 패턴 확인
        galaxy_z_flip_pattern = r'(?:갤럭시|galaxy)\s*(?:Z|z|제트)?\s*(?:flip|플립)\s*(\d+)(?:\s*케이���|\s*강화유리|\s*필름)*'
        galaxy_z_flip_match = re.search(galaxy_z_flip_pattern, text, re.IGNORECASE)
        if galaxy_z_flip_match:
            return f"galaxy Z flip {galaxy_z_flip_match.group(1)}"
        
        galaxy_z_fold_pattern = r'(?:갤럭시|galaxy)\s*(?:Z|z|제트)?\s*(?:fold|폴드)\s*(\d+)(?:\s*케이스|\s*강화유리|\s*필름)*'
        galaxy_z_fold_match = re.search(galaxy_z_fold_pattern, text, re.IGNORECASE)
        if galaxy_z_fold_match:
            return f"galaxy Z fold {galaxy_z_fold_match.group(1)}"
        
        return "unknown"

    def generate_variations(self, base_text, device_type):
        """텍스트 변형 생성"""
        variations = []
        base_text = base_text.lower()
        
        # 기본 변형
        variations.append(base_text)
        
        # 공백 변형
        variations.append(base_text.replace(' ', ''))
        variations.append(re.sub(r'\s+', ' ', base_text))
        
        # 브랜드명 변형
        if device_type in self.variations:
            for prefix in self.variations[device_type]['prefix']:
                for space in self.variations[device_type]['space']:
                    text = base_text
                    if device_type == 'iphone':
                        text = re.sub(r'iphone|아이폰', prefix + space, text, flags=re.IGNORECASE)
                    else:
                        text = re.sub(r'galaxy|갤럭시', prefix + space, text, flags=re.IGNORECASE)
                    variations.append(text)
        
        return list(set(variations))  # 중복 제거
    
    def prepare_training_data(self):
        """학습 데이터 생성"""
        training_data = []
        labels = []
        
        # iPhone 모델 데이터
        iphone_models = {
            '15': ['pro max', 'pro', 'plus', ''],
            '14': ['pro max', 'pro', 'plus', ''],
            '13': ['pro max', 'pro', '']
        }
        
        # Galaxy S 모델 데이터
        galaxy_s_models = {
            '24': ['ultra', 'plus', 'fe', ''],
            '23': ['ultra', 'plus', 'fe', ''],
            '22': ['ultra', 'plus', 'fe', '']
        }
        
        # Galaxy A 모델 데이터
        galaxy_a_models = {
            '54': ['', 'fe'],
            '53': ['', 'fe'],
            '52': ['', 'fe'],
            '34': [''],
            '33': [''],
            '32': ['']
        }
        
        # Galaxy Z 모델 데이터
        galaxy_z_models = {
            'flip': ['6', '5', '4', '3'],
            'fold': ['6', '5', '4', '3']
        }
        
        # 제품 유형
        product_types = [
            '케이스', '강화유리', '필름', '커버', '액세서리',
            '케이스 ', '강화유리 ', '필름 ', '커버 ', '액세서리 ',
            ' 케이스', ' 강화유리', ' 필름', ' 커버', ' 액세서리',
            '투명 케이스', '하드 케이스', '실리콘 케이스', '가죽 케이스',
            '강화유리 필름', '보호 필름', '풀커버 필름'
        ]
        
        # iPhone 데이터 생성
        for number, variants in iphone_models.items():
            for variant in variants:
                base_model = f"iphone {number} {variant}".strip()
                for prefix in self.variations['iphone']['prefix']:
                    for space in self.variations['iphone']['space']:
                        if variant:
                            if variant in self.variations['iphone']['variants']:
                                for var in self.variations['iphone']['variants'][variant]:
                                    model_text = f"{prefix}{space}{number} {var}"
                                    for product_type in product_types:
                                        text = f"{model_text} {product_type}".strip()
                                        training_data.append(text)
                                        labels.append(base_model)
                                        # 순서 변경 변형 추가
                                        text = f"{product_type} {model_text}".strip()
                                        training_data.append(text)
                                        labels.append(base_model)
                        else:
                            model_text = f"{prefix}{space}{number}"
                            for product_type in product_types:
                                text = f"{model_text} {product_type}".strip()
                                training_data.append(text)
                                labels.append(base_model)
                                # 순서 변경 변형 추가
                                text = f"{product_type} {model_text}".strip()
                                training_data.append(text)
                                labels.append(base_model)

        # Galaxy S 데이터 생성
        for number, variants in galaxy_s_models.items():
            for variant in variants:
                base_model = f"galaxy s{number} {variant}".strip()
                for prefix in self.variations['galaxy_s']['prefix']:
                    for space in self.variations['galaxy_s']['space']:
                        if variant:
                            if variant in self.variations['galaxy_s']['variants']:
                                for var in self.variations['galaxy_s']['variants'][variant]:
                                    model_text = f"{prefix}{space}s{number} {var}"
                                    for product_type in product_types:
                                        text = f"{model_text} {product_type}".strip()
                                        training_data.append(text)
                                        labels.append(base_model)
                                        # 순서 변경 변형 추가
                                        text = f"{product_type} {model_text}".strip()
                                        training_data.append(text)
                                        labels.append(base_model)
                        else:
                            model_text = f"{prefix}{space}s{number}"
                            for product_type in product_types:
                                text = f"{model_text} {product_type}".strip()
                                training_data.append(text)
                                labels.append(base_model)
                                # 순서 변경 변형 추가
                                text = f"{product_type} {model_text}".strip()
                                training_data.append(text)
                                labels.append(base_model)

        # Galaxy A 데이터 생성
        for number, variants in galaxy_a_models.items():
            for variant in variants:
                base_model = f"galaxy a{number} {variant}".strip()
                for prefix in self.variations['galaxy_a']['prefix']:
                    for space in self.variations['galaxy_a']['space']:
                        if variant:
                            if variant in self.variations['galaxy_a']['variants']:
                                for var in self.variations['galaxy_a']['variants'][variant]:
                                    model_text = f"{prefix}{space}a{number} {var}"
                                    for product_type in product_types:
                                        text = f"{model_text} {product_type}".strip()
                                        training_data.append(text)
                                        labels.append(base_model)
                                        # 순서 변경 변형 추가
                                        text = f"{product_type} {model_text}".strip()
                                        training_data.append(text)
                                        labels.append(base_model)
                        else:
                            model_text = f"{prefix}{space}a{number}"
                            for product_type in product_types:
                                text = f"{model_text} {product_type}".strip()
                                training_data.append(text)
                                labels.append(base_model)
                                # 순서 변경 변형 추가
                                text = f"{product_type} {model_text}".strip()
                                training_data.append(text)
                                labels.append(base_model)

        # Galaxy Z 데이터 생성
        for form_factor, numbers in galaxy_z_models.items():
            for number in numbers:
                base_model = f"galaxy z {form_factor} {number}"
                for prefix in self.variations['galaxy_z']['prefix']:
                    for space in self.variations['galaxy_z']['space']:
                        if form_factor in self.variations['galaxy_z']['variants']:
                            for var in self.variations['galaxy_z']['variants'][form_factor]:
                                model_text = f"{prefix}{space}z {var} {number}"
                                for product_type in product_types:
                                    text = f"{model_text} {product_type}".strip()
                                    training_data.append(text)
                                    labels.append(base_model)
                                    # 순서 변경 변형 추가
                                    text = f"{product_type} {model_text}".strip()
                                    training_data.append(text)
                                    labels.append(base_model)

        return training_data, labels

    def add_custom_training_data(self, custom_data):
        """사용자 정의 학습 데이터 추가"""
        training_data = []
        labels = []
        
        for text, model in custom_data:
            # 원본 데이터 추가
            training_data.append(text)
            labels.append(model)
            
            # 변형된 데이터도 추가
            normalized_text = self.normalize_model_name(text)
            if normalized_text != text:
                training_data.append(normalized_text)
                labels.append(model)
        
        return training_data, labels

    def update_with_corrections(self, corrections):
        """예측 오류 수정 및 재학습"""
        training_data = []
        labels = []
        
        # 기존 학습 데이터 가져오기
        base_texts, base_labels = self.prepare_training_data()
        training_data.extend(base_texts)
        labels.extend(base_labels)
        
        # 수정 데이터 추가
        for wrong_text, correct_label in corrections:
            # 원본 텍스트 추가
            training_data.append(wrong_text)
            labels.append(correct_label)
            
            # 정규화된 텍스트도 추가
            norm_text = self.normalize_model_name(wrong_text)
            if norm_text != wrong_text:
                training_data.append(norm_text)
                labels.append(correct_label)
        
        # 재학습
        self.train(training_data, labels)

    def train(self, texts=None, labels=None, validate=True):
        """모델 학습"""
        if texts is None or labels is None:
            texts, labels = self.prepare_training_data()
        
        # 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 텍스트 정규화
        X_train_processed = [self.normalize_model_name(text) for text in X_train]
        X_val_processed = [self.normalize_model_name(text) for text in X_val]
        
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
            all_features = self.vectorizer.transform([self.normalize_model_name(text) for text in texts])
            cv_scores = cross_val_score(self.classifier, all_features, labels, cv=5)
            print(f"\n교차 검증 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    def get_bilingual_model_name(self, model_name):
        """모델명의 한글/영문 표기 생성"""
        # 기본 매핑 정의
        mappings = {
            'galaxy': ('갤럭시', 'Galaxy'),
            'iphone': ('아이폰', 'iPhone'),
            'S': ('S', 'S'),      # 대문자로 매핑
            'A': ('A', 'A'),      # 대문자로 매핑
            'Z': ('Z', 'Z'),      # 대문자로 매핑
            'flip': ('플립', 'Flip'),
            'fold': ('폴드', 'Fold'),
            'ultra': ('울트라', 'Ultra'),
            'plus': ('플러스', 'Plus'),
            'pro': ('프로', 'Pro'),
            'max': ('맥스', 'Max'),
            'fe': ('FE', 'FE')
        }
        
        parts = model_name.split()
        kor_parts = []
        eng_parts = []
        
        for part in parts:
            # 시리즈 문자(s, a, z)는 대문자로 변환하여 검색
            if part.lower() in ['s', 'a', 'z']:
                part = part.upper()
                
            if part.lower() in mappings:
                kor, eng = mappings[part.lower()]
                kor_parts.append(kor)
                eng_parts.append(eng)
            else:
                kor_parts.append(part)
                eng_parts.append(part)
        
        kor_name = ' '.join(kor_parts)
        eng_name = ' '.join(eng_parts)
        
        return f"{kor_name} ({eng_name})"

    def predict(self, texts):
        """예측"""
        predictions = []
        
        for text in texts:
            # 먼저 규칙 기반으로 시도
            model_info = self.extract_model_info(text)
            
            if model_info != "unknown":
                # 이중 언어 표기 변환
                bilingual_model = self.get_bilingual_model_name(model_info)
                predictions.append(bilingual_model)
            else:
                # 규칙 기반으로 찾지 못한 경우 머신러닝 모델 사용
                processed_text = self.normalize_model_name(text)
                features = self.vectorizer.transform([processed_text])
                prediction = self.classifier.predict(features)[0]
                # 이중 언어 표기 변환
                bilingual_model = self.get_bilingual_model_name(prediction)
                predictions.append(bilingual_model)
            
            # 예측 결과 로깅
            self.log_prediction(text, predictions[-1])
        
        return predictions

    def log_prediction(self, input_text, predicted, actual=None):
        """예측 결과 로깅"""
        self.prediction_history.append({
            'input': input_text,
            'predicted': predicted,
            'actual': actual,
            'timestamp': datetime.now()
        })

    def analyze_errors(self):
        """예측 오류 분석"""
        errors = [p for p in self.prediction_history if p['actual'] and p['predicted'] != p['actual']]
        if errors:
            print("\n예측 오류 분석:")
            for error in errors:
                print(f"\n입력: {error['input']}")
                print(f"예측: {error['predicted']}")
                print(f"실제: {error['actual']}")
    
    def save_model(self, directory="models"):
        """모델 저장"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(directory, f"phone_model_classifier_{timestamp}")
        os.makedirs(model_path)
        
        # 모델 저장
        import joblib
        joblib.dump(self.classifier, os.path.join(model_path, "classifier.pkl"))
        joblib.dump(self.vectorizer, os.path.join(model_path, "vectorizer.pkl"))
        
        # 설정 저장
        config = {
            "model_patterns": self.model_patterns,
            "variations": self.variations,
            "prediction_history": [
                {
                    'input': p['input'],
                    'predicted': p['predicted'],
                    'actual': p['actual'],
                    'timestamp': p['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                }
                for p in self.prediction_history
            ]
        }
        
        with open(os.path.join(model_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        print(f"모델이 저장되었습니다: {model_path}")

    @classmethod
    def load_model(cls, model_path):
        """모델 로드"""
        import joblib
        
        instance = cls()
        
        # 모델 로드
        instance.classifier = joblib.load(os.path.join(model_path, "classifier.pkl"))
        instance.vectorizer = joblib.load(os.path.join(model_path, "vectorizer.pkl"))
        
        # 설정 로드
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
            instance.model_patterns = config["model_patterns"]
            instance.variations = config["variations"]
            
            # 예측 이력 로드
            instance.prediction_history = [
                {
                    'input': p['input'],
                    'predicted': p['predicted'],
                    'actual': p['actual'],
                    'timestamp': datetime.strptime(p['timestamp'], "%Y-%m-%d %H:%M:%S")
                }
                for p in config.get("prediction_history", [])
            ]
            
        return instance

def main():
    # 모델 파일이 저장된 경로 지정
    MODEL_DIR = "models"
    
    # 가장 최근의 모델 폴더 찾기
    latest_model = None
    if os.path.exists(MODEL_DIR):
        model_folders = [
            os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR)
            if d.startswith("phone_model_classifier_")
        ]
        if model_folders:
            latest_model = max(model_folders, key=os.path.getctime)
    
    try:
        if latest_model:
            # 저장된 모델 불러오기 시도
            print(f"저장된 모델을 불러오는 중... ({latest_model})")
            classifier = PhoneModelClassifier.load_model(latest_model)
            print("모델 로드 완료!")
        else:
            raise FileNotFoundError("저장된 모델을 찾을 수 없습니다.")
            
    except (FileNotFoundError, OSError) as e:
        # 저장된 모델이 없는 경우 새로 학습
        print(f"저장된 모델이 없습니다. 새로운 모델을 학습합니다... (사유: {str(e)})")
        classifier = PhoneModelClassifier()
        classifier.train()
        
        # 학습된 모델 저장
        print("학습된 모델을 저장합니다...")
        classifier.save_model(MODEL_DIR)
        print("모델 저장 완료!")

    # 테스트
    test_texts = [
        "맥씨 강화유리 필름 (5매) / iPhone 16 Plus / 아이폰 16 플러스",
        "크레이지 범퍼 케이스 / SM-S721 / 갤럭시 S24 FE",
        "DC 투카드 맥세이프 범퍼 케이스 / SM-S928 / 갤럭시 S24 울트라",
        "프리모 방탄톡 젤리 케이스 / SM-S721 / 갤럭시 S24 FE",
        "내셔널지오그래픽 마그네틱 힌지 커버 슬림 스탠드 Z폴드 케이스 (with S펜 홀더) / SM-F956 / 갤럭시 Z폴드6",
        "187. 내셔널지오그래픽 마그네틱 힌지 커버 슬림 스탠드 Z폴드 케이스 (with S펜 홀더) / SM-F946 / 갤럭시 Z폴드5",
        "299. 감성 다이어리 케이스 / SM-A336 / 갤럭시 A33 (5G)",
        "303. 세븐 포켓 다이어리 케이스 / SM-A155N / 갤럭시 A15 LTE / SM-A156L / 갤럭시 A15 (5G) ★공용★",
        "003._베온_초강력_맥세이프_클리어_케이스__iPhone_16_Plus__아이폰_16_플러스"
    ]
    
    predictions = classifier.predict(test_texts)
    for text, pred in zip(test_texts, predictions):
        print(f"\n입력: {text}")
        print(f"예측: {pred}")

if __name__ == "__main__":
    main()
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import requests
import csv
import re
import base64
import urllib3
import logging
from phone_model_deep_filter import PhoneModelClassifier
from category_classifier import CategoryClassifier
from product_csv_handler import ProductCSVHandler
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 로깅 설정 추가 (파일 상단에 추가)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 콘솔에 출력
    ]
)
logger = logging.getLogger(__name__)

class CrossmallAutomation:
    def __init__(self):
        logger.info("크로스몰 자동화 초기화")
        # Chrome 웹드라이버 옵션 설정
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--start-maximized')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        
        # 다중 파일 다운로드 관련 설정 추가
        self.options.add_experimental_option('prefs', {
            'profile.default_content_setting_values.automatic_downloads': 1,
            'download.prompt_for_download': False,
            'download.directory_upgrade': True,
            'safebrowsing.enabled': True
        })
        
        # 웹드라이버 초기화 (자동 설치 포함)
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=self.options)
        self.wait = WebDriverWait(self.driver, 10)  # 최대 10초 대기

    def login(self, username, password):
        """크로스몰 로그인 함수"""
        try:
            # 로그인 페이지 접속
            login_url = "https://www.crossmall.co.kr/mall/m_login.php?url=/mall/index.php&ps_db=&ps_db=&ps_boid=&ps_pname=&ps_goid=&ps_ctid=&ps_page="
            self.driver.get(login_url)
            
            # 페이지 로딩 대기
            time.sleep(2)

            # ID 입력
            id_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/form/table/tbody/tr/td/table[1]/tbody/tr/td[2]/input"
            id_field = self.wait.until(EC.presence_of_element_located((By.XPATH, id_xpath)))
            id_field.clear()
            id_field.send_keys(username)

            # 비밀번호 입력
            pw_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/form/table/tbody/tr/td/table[1]/tbody/tr/td[4]/input"
            pw_field = self.wait.until(EC.presence_of_element_located((By.XPATH, pw_xpath)))
            pw_field.clear()
            pw_field.send_keys(password)

            # 로그인 버튼 클릭 (JavaScript 실행으로 변경)
            login_button_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/form/table/tbody/tr/td/table[1]/tbody/tr/td[5]/input"
            login_button = self.wait.until(EC.presence_of_element_located((By.XPATH, login_button_xpath)))
            
            # JavaScript로 클릭 실행
            self.driver.execute_script("arguments[0].click();", login_button)

            # 로그인 성공 여부 확인
            time.sleep(3)  # 대기 시간 증가
            
            if "index.php" in self.driver.current_url:
                logger.info("로그인 성공!")
                return True
            
            if "로그인" in self.driver.page_source or "아이디" in self.driver.page_source:
                logger.error("로그인 실패: 아이디 또는 비밀번호가 올바르지 않습니다.")
                return False
                
            return True

        except TimeoutException as e:
            logger.error(f"로그인 시간 초과: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"로그인 중 오류 발생: {str(e)}")
            return False

    def close(self):
        """브라우저 종료"""
        if self.driver:
            self.driver.quit()


class CrossmallCategoryCrawler:
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(self.driver, 10)
        self.product_counter = 1  # 추가
        self.visited_categories = set()
        self.visited_products = set()
        self.current_main_category = None
        self.current_sub_category = None
        
        # CSV 핸들러 초기화
        self.csv_handler = ProductCSVHandler()
        
        # 휴대폰 모델 분류기 초기화
        try:
            print("휴대폰 모델 분류기 초기화 중...")
            MODEL_DIR = "models"
            if os.path.exists(MODEL_DIR):
                model_folders = [
                    os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR)
                    if d.startswith("phone_model_classifier_")
                ]
                if model_folders:
                    latest_model = max(model_folders, key=os.path.getctime)
                    print(f"저장된 모델을 불러오는 중... ({latest_model})")
                    self.model_classifier = PhoneModelClassifier.load_model(latest_model)
                    print("휴대폰 모델 분류기 로드 완료!")
                else:
                    print("저장된 모델이 없어 새로 학습합니다...")
                    self.model_classifier = PhoneModelClassifier()
                    self.model_classifier.train()
            else:
                print("모델 디렉토리가 없어 새로 학습합니다...")
                self.model_classifier = PhoneModelClassifier()
                self.model_classifier.train()
        except Exception as e:
            print(f"모델 분류기 초기화 실패: {str(e)}")
            self.model_classifier = None

        # 카테고리 분류기 초기화
        try:
            print("카테고리 분류기 초기화 중...")
            CATEGORY_MODEL_DIR = "category_classifier"
            if os.path.exists(CATEGORY_MODEL_DIR):
                model_folders = [
                    os.path.join(CATEGORY_MODEL_DIR, d) for d in os.listdir(CATEGORY_MODEL_DIR)
                    if d.startswith("model_")
                ]
                if model_folders:
                    latest_model = max(model_folders, key=os.path.getctime)
                    print(f"저장된 모델을 불러오는 중... ({latest_model})")
                    self.category_classifier = CategoryClassifier.load_model(latest_model)
                    print("카테고리 분류기 로드 완료!")
                else:
                    print("저장된 모델이 없어 새로 학습합니다...")
                    self.category_classifier = CategoryClassifier()
                    self.category_classifier.train()
            else:
                print("모델 디렉토리가 없어 새로 학습합니다...")
                self.category_classifier = CategoryClassifier()
                self.category_classifier.train()
        except Exception as e:
            print(f"카테고리 분류기 초기화 실패: {str(e)}")
            self.category_classifier = None

    def sanitize_filename(self, filename):
        """파일명에서 특수문자 제거"""
        # 윈도우 파일시스템에서 사용할 수 없는 문자 제거
        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        # 공백을 언더스코어로 변경
        filename = filename.replace(" ", "_")
        return filename

    def save_product_info(self, product_info):
        """상품 정보를 CSV 파일에 저장"""
        try:
            base_price = int(product_info['price'])
            selling_price = round(base_price * 1.5, -1)
            
            # CSV 형식으로 데이터 구성
            row_data = {
                'category_id': product_info.get('category_id', ''),
                'name': product_info.get('name', ''),
                'display_name': product_info.get('name', ''),
                'option_mandatory': product_info.get('option_mandatory', ''),
                'option_mix': product_info.get('option_mix', '미조합'),
                'option_name': product_info.get('option_name', ''),
                'option_values': product_info.get('option_values', ''),
                'option_prices': product_info.get('option_prices', ''),
                'opt_use': product_info.get('opt_use', ''),
                'opt_oneclick': product_info.get('opt_oneclick', ''),
                'option_type': product_info.get('option_type', '분리형'),
                'selling_price': str(selling_price),
                'stock': product_info.get('stock', '100'),
                'consumer_price': str(selling_price),
                'supply_price': str(base_price),
                'point_percentage': product_info.get('point_percentage', '10'),
                'reserve': product_info.get('reserve', ''),
                'mobile_reserve': product_info.get('mobile_reserve', ''),
                'point': product_info.get('point', ''),
                'search_tags': product_info.get('search_tags', ''),
                'adult_auth': product_info.get('adult_auth', 'N'),
                'display_status': product_info.get('display_status', '진열'),
                'main_image': product_info.get('main_image', ''),
                'detail_image': product_info.get('detail_image', 'AUTO'),
                'list_image': product_info.get('list_image', 'AUTO'),
                'mobile_image': product_info.get('mobile_image', ''),
                'description': product_info.get('description', ''),
                'mobile_description': product_info.get('mobile_description', ''),
                'model_name': product_info.get('model_name', ''),
                'best_product_display': product_info.get('best_product_display', 'N'),
                'vat_type': product_info.get('vat_type', 'Y')
            }
            
            return self.csv_handler.append_product(row_data)
            
        except Exception as e:
            self.logger.error(f"상품 정보 저장 중 오류 발생: {str(e)}")
            return False

    def navigate_to_category_page(self):
        """메인 카테고리 페이지로 이동"""
        try:
            category_url = "https://www.crossmall.co.kr/mall/m_mall_list.php?ps_ctid=64000000"
            self.driver.get(category_url)
            time.sleep(2)  # 페이지 로딩 대기
            print("카테고리 페이지 이동 성공")
        except Exception as e:
            print(f"카테고리 페이지 이동 중 오류: {str(e)}")
            raise

    def get_main_categories(self):
        """메인 카테고리(select 옵션) 목록 가져오기"""
        try:
            select_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select[onchange='select_link(this.value)']"))
            )
            select_object = Select(select_element)
            categories = select_object.options
            print(f"메인 카테고리 {len(categories)}개 발견")
            return categories
        except Exception as e:
            print(f"메인 카테고리 목록 가져오기 실패: {str(e)}")
            return []

    def get_sub_categories(self):
        """서브 카테고리(a 태그) 목록 가져오기"""
        try:
            base_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/table[1]"
            subcategories = self.wait.until(
                EC.presence_of_all_elements_located((By.XPATH, f"{base_xpath}//a"))
            )
            valid_subcategories = [
                sub for sub in subcategories 
                if sub.get_attribute('href') and 'm_mall_list.php' in sub.get_attribute('href')
            ]
            print(f"서브 카테고리 {len(valid_subcategories)}개 발견")
            return valid_subcategories
        except Exception as e:
            print(f"서브 카테고리 목록 져오기 실패: {str(e)}")
            return []

    def process_categories(self):
        """모든 카테고리를 순회하며 처리"""
        main_categories = self.get_main_categories()
        total_categories = len(main_categories)
        
        for index, main_category in enumerate(main_categories, 1):
            try:
                category_value = main_category.get_attribute('value')
                if category_value in self.visited_categories:
                    print(f"이미 처리된 카테고리 스킵: {main_category.text}")
                    continue

                print(f"\n[{index}/{total_categories}] 메인 카테고리 처리 중: {main_category.text}")
                self.current_main_category = main_category.text
                
                # 메인 카테고리 선택
                select_element = Select(self.driver.find_element(
                    By.CSS_SELECTOR, "select[onchange='select_link(this.value)']"
                ))
                select_element.select_by_value(category_value)
                time.sleep(2)  # 페이지 로딩 대기

                # 방문 기록 추가
                self.visited_categories.add(category_value)

                # 서브 카테고리 처리
                self.process_sub_categories()

            except Exception as e:
                print(f"카테고리 처리 중 오류 발생: {str(e)}")
                continue

    def process_sub_categories(self):
        """현재 페이지의 서브 카테고리 처리"""
        sub_categories = self.get_sub_categories()
        total_subs = len(sub_categories)
        
        for index, sub_category in enumerate(sub_categories, 1):
            try:
                sub_url = sub_category.get_attribute('href')
                if not sub_url or sub_url in self.visited_categories:
                    continue

                print(f"[{index}/{total_subs}] 서브 카테고리 처리 중: {sub_category.text}")
                self.current_sub_category = sub_category.text
                
                # 새 탭에서 서브 카테고리 열기
                self.driver.execute_script("window.open(arguments[0]);", sub_url)
                time.sleep(1)  # 페이지 로딩 대기
                
                # 새 탭으로 전환
                self.driver.switch_to.window(self.driver.window_handles[-1])
                
                # 상품 목록 페이지 처리
                self.process_product_list_page()
                
                # 방문 기록 추가
                self.visited_categories.add(sub_url)
                
                # 탭 닫고 원래 탭으로 복귀
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[0])
                
                # 과도한 요청 방지를 위한 딜레이
                time.sleep(1.5)
                
            except Exception as e:
                print(f"서브 카테고리 처리 중 오류 발생: {str(e)}")
                # 에러 발생 시 원래 탭으로 복귀 시도
                if len(self.driver.window_handles) > 1:
                    self.driver.close()
                    self.driver.switch_to.window(self.driver.window_handles[0])
                continue

    def process_product_list_page(self):
        """상품 목록 페이지에서 각 상품 처리"""
        page_number = 1
        
        while True:  # 페이징 처리를 위 루프
            try:
                print(f"\n현재 페이지: {page_number}")
                
                # 상품 목록 테이블 찾기
                product_table_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/table[5]/tbody"
                product_table = self.wait.until(
                    EC.presence_of_element_located((By.XPATH, product_table_xpath))
                )

                # 상품 링크 찾기
                product_links = product_table.find_elements(
                    By.CSS_SELECTOR, "tr > td.tb_3 > div.goods_name_ > a"
                )

                if not product_links:
                    print("상품이 없거나 더 상의 상품이 없습니다.")
                    break

                total_products = len(product_links)
                print(f"현재 페이지에서 {total_products}개의 상품을 발견했습니다.")

                # 각 상품 처리
                for index, product_link in enumerate(product_links, 1):
                    try:
                        product_url = product_link.get_attribute('href')
                        
                        # 이미 처리한 상품 스킵
                        if product_url in self.visited_products:
                            print(f"이미 처리된 상품 스킵: {product_link.text.strip()}")
                            continue
                            
                        product_name = product_link.text.strip()
                        print(f"[{index}/{total_products}] 상품 처리 중: {product_name}")

                        # 새 탭에서 상품 페이지 열기
                        self.driver.execute_script("window.open(arguments[0]);", product_url)
                        time.sleep(1)  # 페이지 로딩 대기
                        
                        # 상품 페이지 탭으로 전환
                        self.driver.switch_to.window(self.driver.window_handles[-1])

                        # 상품 정보 수집
                        self.process_product_page(product_name)

                        # 방문 기록 추가
                        self.visited_products.add(product_url)

                        # 상품 페이지 탭 닫기
                        self.driver.close()
                        self.driver.switch_to.window(self.driver.window_handles[-1])

                        # 과도한 요청 방지를 위한 딜레이
                        time.sleep(1)

                    except Exception as e:
                        print(f"상품 처 중 오류 발생: {str(e)}")
                        # 에러 발생 시 원래 탭으로 복귀
                        if len(self.driver.window_handles) > 1:
                            self.driver.close()
                            self.driver.switch_to.window(self.driver.window_handles[-1])
                        continue

                # 다음 페이지 확인 및 이동
                try:
                    next_page = self.driver.find_element(
                        By.XPATH, "//a[contains(text(), '[다음]')]"
                    )
                    next_page.click()
                    time.sleep(1.5)  # 페이지 로딩 대기
                    page_number += 1
                except NoSuchElementException:
                    print("마지막 페이지입니다.")
                    break
                except Exception as e:
                    print(f"페이지 이동 중 오류 발생: {str(e)}")
                    break

            except Exception as e:
                print(f"상품 목록 처리 중 오류 발생: {str(e)}")
                break

    def extract_phone_model(self, product_name):
        """상품명에서 휴대폰 모델명 추출"""
        try:
            if self.model_classifier:
                predictions = self.model_classifier.predict([product_name])
                if predictions and predictions[0] != "unknown":
                    return predictions[0]
            return None
        except Exception as e:
            print(f"모델명 추출 중 오류 발생: {str(e)}")
            return None

    def classify_product_category(self, product_name):
        """상품명으로 카테고리 분류"""
        try:
            if self.category_classifier:
                predictions = self.category_classifier.predict([product_name])
                if predictions and len(predictions) > 0:
                    return predictions[0]['category_id']  # 카테고리 ID만 반환
            return None
        except Exception as e:
            print(f"카테고리 분류 중 오류 발생: {str(e)}")
            return None

    def process_product_page(self, product_name):
        try:
            # 페이지 로�� 대기
            time.sleep(1)

            # 상품명 추출
            try:
                product_name_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/form/table[1]/tbody/tr/td[2]/div[1]"
                product_name_element = self.wait.until(EC.presence_of_element_located((By.XPATH, product_name_xpath)))
                actual_product_name = product_name_element.text.strip()
                print(f"\n상품명: {actual_product_name}")

                # 상품명에서 앞의 숫자와 점 제거
                def clean_product_name(name):
                    # 숫자. 또는 숫자_ 패턴 제거
                    cleaned_name = re.sub(r'^\d+[\._]\s*', '', name)
                    return cleaned_name.strip()
                
                # 상품명 정제
                actual_product_name = clean_product_name(actual_product_name)
                print(f"\n정제된 상품명: {actual_product_name}")

                # 이미 저장된 상품인지 확인
                if actual_product_name in self.csv_handler.get_saved_products():
                    logger.info(f"이미 저장된 상품 스킵: {actual_product_name}")
                    return

            except Exception as e:
                print(f"상품명 추출 중 오류: {str(e)}")
                actual_product_name = product_name

            # 이미지 처리 관련 JavaScript 실행
            try:
                # 메인 이미지 다운로드 스크립트
                main_image_script = """
                window.finish = false;
                var img = $("body > table > tbody > tr > td > table:nth-child(6) > tbody > tr > td > form > table:nth-child(7) > tbody > tr > td:nth-child(1) > table > tbody > tr:nth-child(1) > td img");
                var mainImageUrl = img.attr('src');
                return mainImageUrl;
                """
                main_image_url = self.driver.execute_script(main_image_script)

                # 상세 이미지 URL 가져오기 스크립트
                detail_images_script = """
                var images = $("body > table > tbody > tr > td > table:nth-child(6) > tbody > tr > td > form > div:nth-child(12) img");
                var urls = [];
                images.each(function(i, image) {
                    var imageUrl = $(image).attr('src');
                    if (!imageUrl.endsWith('.gif')) {
                        urls.push(imageUrl);
                    }
                });
                return urls;
                """
                detail_image_urls = self.driver.execute_script(detail_images_script)

                # 이미지 저장을 위한 디렉토리 생성
                base_dir = "product_images"
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)

                # 제품 번호 생성 (6자리 숫자)
                product_number = str(self.product_counter).zfill(6)
                
                # 이미지 경로 설정
                image_base_path = f"product_images/{product_number}"
                os.makedirs(image_base_path, exist_ok=True)

                # 이미지 URL과 파일 경로 저장을 위한 변수
                saved_main_image_path = ''
                saved_detail_image_paths = []

                # 메인 이미지 다운로드
                if main_image_url:
                    main_image_filename = f"{product_number}_M.jpg"
                    target_path = os.path.join(image_base_path, main_image_filename)
                    
                    if self.download_and_move_image(main_image_url, main_image_filename, target_path):
                        saved_main_image_path = f"product_images/{product_number}/{main_image_filename}"
                        print(f"메인 이미지 경로 저장: {saved_main_image_path}")
                    time.sleep(1)  # 다음 이미지 처리 전 대기

                # 상세 이미지 다운로드
                if detail_image_urls:
                    for idx, url in enumerate(detail_image_urls, 1):
                        detail_image_filename = f"{product_number}_{str(idx).zfill(2)}.jpg"
                        target_path = os.path.join(image_base_path, detail_image_filename)
                        
                        if self.download_and_move_image(url, detail_image_filename, target_path):
                            saved_path = f"product_images/{product_number}/{detail_image_filename}"
                            saved_detail_image_paths.append(saved_path)
                            print(f"상세 이미지 경로 저장: {saved_path}")
                        time.sleep(1)  # 다음 이미지 처리 전 대기

            except Exception as e:
                print(f"이미지 처리 중 오류: {str(e)}")
                main_image_url = None

            # 옵션 정보 추출 (수정된 부분)
            options_text = ""
            try:
                option_list = self.driver.find_element(By.CLASS_NAME, "option_list")
                if option_list:
                    option_text_div = option_list.find_element(By.CLASS_NAME, "option_text")
                    if option_text_div:
                        options_text = option_text_div.text.strip()
                        # '|' 구분자로 분해하고 공백 제거 후 다시 결합
                        options = [opt.strip() for opt in options_text.split('|') if opt.strip()]
                        options_text = ' | '.join(options)
                        print(f"상품 옵션: {options_text}")
            except NoSuchElementException:
                print("옵션 정보가 없는 상품입니다.")
                options_text = "단일상품"
            except Exception as e:
                print(f"옵션 정보 추출 중 오류: {str(e)}")
                options_text = "단일상품"

            # 가격 추출
            option_price_value = None
            try:
                option_price_element = self.driver.find_element(By.NAME, "option_money")
                if option_price_element:
                    option_price_value = option_price_element.get_attribute("value")
                    option_price_value = re.sub(r'[^\d]', '', option_price_value)
                    print(f"옵션 가격: {option_price_value}")
            except Exception as e:
                print(f"가격 추출 실패: {str(e)}")

            # 상품명에서 휴대폰 모델 추출
            phone_model = self.extract_phone_model(actual_product_name)
            if phone_model:
                print(f"추출된 휴대폰 모델: {phone_model}")

            # 카테고리 분류
            category_id = self.classify_product_category(actual_product_name)

            # 상품 정보 저장
            try:
                base_price = int(option_price_value) if option_price_value else 0
                selling_price = round(base_price * 1.5, -2)  # 1.5배 가격, 100원 단위 반올림

                # 제품 번호 생성 (6자리 숫자)
                product_number = str(self.product_counter).zfill(6)
                
                # 이미지 경로 설정
                image_base_path = f"product_images/{product_number}"
                os.makedirs(image_base_path, exist_ok=True)

                # 이미지 URL과 파일 경로 저장을 위한 변수
                saved_main_image_path = ''
                saved_detail_image_paths = []

                # 메인 이미지 다운로드
                if main_image_url:
                    main_image_filename = f"{product_number}_M.jpg"
                    target_path = os.path.join(image_base_path, main_image_filename)
                    
                    if self.download_and_move_image(main_image_url, main_image_filename, target_path):
                        saved_main_image_path = f"product_images/{product_number}/{main_image_filename}"
                        print(f"메인 이미지 경로 저장: {saved_main_image_path}")
                    time.sleep(1)  # 다음 이미지 처리 전 대기

                # 상세 이미지 다운로드
                if detail_image_urls:
                    for idx, url in enumerate(detail_image_urls, 1):
                        detail_image_filename = f"{product_number}_{str(idx).zfill(2)}.jpg"
                        target_path = os.path.join(image_base_path, detail_image_filename)
                        
                        if self.download_and_move_image(url, detail_image_filename, target_path):
                            saved_path = f"product_images/{product_number}/{detail_image_filename}"
                            saved_detail_image_paths.append(saved_path)
                            print(f"상세 이미지 경로 저장: {saved_path}")
                        time.sleep(1)  # 다음 이미지 처리 전 대기

                # 상품 정보 구성
                product_info = {
                    'category_id': category_id or '',
                    'name': actual_product_name,
                    'display_name': actual_product_name,
                    'option_mandatory': '필수' if options_text != '단일상품' else '',
                    'option_mix': '미조합',
                    'option_name': '기타' if options_text != '단일상품' else '',
                    'option_values': options_text if options_text != '단일상품' else '',
                    'option_prices': ','.join(['0'] * len(options_text.split('|'))) if options_text != '단일상품' else '',
                    'opt_use': '사용' if options_text != '단일상품' else '',
                    'opt_oneclick': '사용' if options_text != '단일상품' else '',
                    'option_type': '선택형' if options_text != '단일상품' else '',
                    'price': base_price,
                    'stock': '100',
                    'selling_price': str(selling_price),
                    'consumer_price': str(selling_price),
                    'supply_price': str(base_price),
                    'point_percentage': '10',
                    'main_image': saved_main_image_path,
                    'detail_image': ','.join(saved_detail_image_paths) if saved_detail_image_paths else 'AUTO',
                    'list_image': saved_main_image_path,
                    'mobile_image': saved_main_image_path,
                    'description': f"<!--[OPENEDITOR]--><p>{actual_product_name}</p>" + \
                                 ''.join([f"<img src=\"{path}\" />" for path in [saved_main_image_path] + saved_detail_image_paths]) \
                                 if saved_main_image_path else '',
                    'mobile_description': f"<!--[OPENEDITOR]--><p>{actual_product_name}</p>" + \
                                       ''.join([f"<img src=\"{path}\" />" for path in [saved_main_image_path] + saved_detail_image_paths]) \
                                       if saved_main_image_path else '',
                    'display_status': '진열',
                    'adult_auth': 'N',
                    'best_product_display': 'N',
                    'vat_type': 'Y'
                }

                # 상품 정보 저장
                if self.save_product_info(product_info):
                    self.product_counter += 1
                    return True
                return False

            except Exception as e:
                print(f"❌ 상품 정보 처리 중 오류: {str(e)}")
                logger.error(f"상품 정보 처리 중 오류 발생: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"상품 페이지 처리 중 오류 발생: {str(e)}")
            return False
        finally:
            print("-" * 80)

    def download_and_move_image(self, image_url, filename, target_path):
        """이미지 다운로드 및 이동 처리"""
        try:
            # 이미 대상 경로에 파일이 있는 경우 삭제
            if os.path.exists(target_path):
                os.remove(target_path)
                time.sleep(0.5)  # 파일 삭제 후 대기
            
            # 다운로드 경로의 파일도 확인 및 삭제
            downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
            downloaded_file = os.path.join(downloads_path, filename)
            if os.path.exists(downloaded_file):
                os.remove(downloaded_file)
                time.sleep(0.5)  # 파일 삭제 후 대기

            # 이미지 다운로드
            download_script = """
            var link = $('<a>');
            link.attr('href', arguments[0]);
            link.attr('download', arguments[1]);
            $('body').append(link);
            link[0].click();
            link.remove();
            return true;
            """
            self.driver.execute_script(download_script, image_url, filename)
            
            # 다운로드 완료 대기
            max_wait = 10  # 최대 10초 대기
            while max_wait > 0:
                if os.path.exists(downloaded_file):
                    time.sleep(1)  # 다운로드 완료 후 추가 대기
                    break
                time.sleep(1)
                max_wait -= 1
            
            # 파일 이동
            if os.path.exists(downloaded_file):
                time.sleep(1)  # 이동 전 추가 대기
                os.rename(downloaded_file, target_path)
                time.sleep(1)  # 이동 후 추가 대기
                print(f"이미지 저장 성공: {target_path}")
                return True
            
            print(f"이미지 다운로드 실패: {filename}")
            return False
            
        except Exception as e:
            print(f"이미지 다운로드 중 오류: {str(e)}")
            return False

def main():
    automation = CrossmallAutomation()
    
    try:
        # 환경 변수에서 로그인 정보 가져오기 (또는 직접 입력)
        username = os.getenv('CROSSMALL_ID', '7530ldc')
        password = os.getenv('CROSSMALL_PW', 'lbc25802580')
        
        # 로그인 시도
        login_success = automation.login(username, password)
        
        if login_success:
            # 카테고리 크롤러 초기화 및 실행
            crawler = CrossmallCategoryCrawler(automation.driver)
            crawler.navigate_to_category_page()
            crawler.process_categories()
        else:
            print("로그인 실패로 프로그램을 종료합니��.")

    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")
    
    finally:
        # 작업 완료 후 브라우저 종료
        automation.close()

if __name__ == "__main__":
    main()
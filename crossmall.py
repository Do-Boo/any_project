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

            # ID 입력
            id_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/form/table/tbody/tr/td/table[1]/tbody/tr/td[2]/input"
            id_field = self.wait.until(EC.presence_of_element_located((By.XPATH, id_xpath)))
            id_field.clear()  # 기존 입력값 제거
            id_field.send_keys(username)

            # 비밀번호 입력
            pw_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/form/table/tbody/tr/td/table[1]/tbody/tr/td[4]/input"
            pw_field = self.wait.until(EC.presence_of_element_located((By.XPATH, pw_xpath)))
            pw_field.clear()  # 기존 입력값 제거
            pw_field.send_keys(password)

            # 로그인 버튼 클릭
            login_button_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/form/table/tbody/tr/td/table[1]/tbody/tr/td[5]/input"
            login_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, login_button_xpath)))
            login_button.click()

            # 로그인 성공 여부 확인
            time.sleep(2)  # 페이지 전환 대기
            
            # URL로 로그인 성공 확인
            if "index.php" in self.driver.current_url:
                logger.info("로그인 성공!")
                return True
            
            # 추가적인 로그인 실패 확인
            if "로그인" in self.driver.page_source or "아이디" in self.driver.page_source:
                logger.error("로그인 실패: 아이디 또는 비밀번호가 올바르지 않습니다.")
                return False
                
            return True

        except TimeoutException as e:
            print(f"로그인 시간 초과: {str(e)}")
            return False
        except Exception as e:
            print(f"로그인 중 오류 발생: {str(e)}")
            return False

    def close(self):
        """브라우저 종료"""
        if self.driver:
            self.driver.quit()


class CrossmallCategoryCrawler:
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(self.driver, 10)
        self.visited_categories = set()  # 이미 방문한 카테고리 URL 저장
        self.visited_products = set()  # 이미 방문한 상품 URL 저장
        self.current_main_category = None
        self.current_sub_category = None

    def sanitize_filename(self, filename):
        """파일명에서 특수문자 제거"""
        # 윈도우 파일시스템에서 사용할 수 없는 문자 제거
        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        # 공백을 언더스코어로 변경
        filename = filename.replace(" ", "_")
        return filename

    def save_product_info(self, product_info):
        """수집된 상품 정보를 CSV 파일로 저장"""
        try:
            filename = 'products.csv'
            file_exists = os.path.isfile(filename)
            
            with open(filename, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=product_info.keys())
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(product_info)
                
            print(f"상품 정보 CSV 저장 완료: {product_info['name']}")
            
        except Exception as e:
            print(f"상품 정보 CSV 저장 중 오류 발생: {str(e)}")

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
            print(f"서브 카테고리 목록 가져오기 실패: {str(e)}")
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
        
        while True:  # 페이징 처리를 위한 루프
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
                    print("상품이 없거나 더 ��상의 상품이 없습니다.")
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
                        print(f"상품 처리 중 오류 발생: {str(e)}")
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

    def process_product_page(self, product_name):
        """개별 상품 페이지에서 정보 수집"""
        try:
            # 페이지 로딩 대기
            time.sleep(1)

            # 상품명 추출
            try:
                product_name_xpath = "/html/body/table/tbody/tr/td/table[4]/tbody/tr/td/form/table[1]/tbody/tr/td[2]/div[1]"
                product_name_element = self.wait.until(EC.presence_of_element_located((By.XPATH, product_name_xpath)))
                actual_product_name = product_name_element.text.strip()
                print(f"\n상품명: {actual_product_name}")
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
                
                safe_product_name = self.sanitize_filename(actual_product_name)
                save_dir = os.path.join(base_dir, safe_product_name)
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 메인 이미지 다운로드 실행
                if main_image_url:
                    download_script = """
                    var link = $('<a>');
                    link.attr('href', arguments[0]);
                    link.attr('download', arguments[1]);
                    $('body').append(link);
                    link[0].click();
                    link.remove();
                    return true;
                    """
                    main_image_filename = f"{safe_product_name}_M.jpg"
                    self.driver.execute_script(download_script, main_image_url, main_image_filename)
                    time.sleep(1)  # 다운로드 대기

                    # 다운로드된 파일 이동
                    downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
                    downloaded_file = os.path.join(downloads_path, main_image_filename)
                    if os.path.exists(downloaded_file):
                        target_path = os.path.join(save_dir, main_image_filename)
                        os.rename(downloaded_file, target_path)
                        print(f"메인 이미지 저장 완료: {target_path}")
                    else:
                        print("메인 이미지 다운로드 실패")
                    time.sleep(1)  # 추가 딜레이

                # 상세 이미지 다운로드
                if detail_image_urls:
                    print(f"\n총 {len(detail_image_urls)}개의 상세 이��지 발견")
                    for idx, url in enumerate(detail_image_urls, 1):
                        try:
                            detail_image_filename = f"{safe_product_name}_{str(idx).zfill(2)}.jpg"
                            self.driver.execute_script(download_script, url, detail_image_filename)
                            time.sleep(1)  # 다운로드 대기

                            # 다운로드된 파일 이동
                            downloaded_file = os.path.join(downloads_path, detail_image_filename)
                            if os.path.exists(downloaded_file):
                                target_path = os.path.join(save_dir, detail_image_filename)
                                os.rename(downloaded_file, target_path)
                                print(f"상세 이미지 {idx} 저장 완료: {target_path}")
                            else:
                                print(f"상세 이미지 {idx} 다운로드 실패")
                            time.sleep(1)  # 추가 딜레이
                        except Exception as e:
                            print(f"상세 이미지 {idx} 처리 중 오류: {str(e)}")

            except Exception as e:
                print(f"이미지 처리 중 오류: {str(e)}")

            # 옵션 정보 추출  
            options_list = []
            try:
                option_elements = self.driver.find_elements(By.CLASS_NAME, "option_text")
                if option_elements:
                    print("\n옵션 목록:")
                    for option in option_elements:
                        option_text = option.text.strip()
                        options_list.append(option_text)
                        print(f"- {option_text}")
            except Exception as e:
                print("옵션 정보 없거나 추출 실패")

            # 옵션 가격 추출
            option_price_value = None
            try:
                option_price = self.driver.find_element(By.NAME, "option_money").get_attribute("value")
                if option_price:
                    option_price_value = option_price
                    print(f"옵션 가격: {option_price}")
            except Exception as e:
                print("옵션 가격 정보가 없거나 추출 실패")

            # 상품 정보 구성 및 저장
            product_info = {
                'url': self.driver.current_url,
                'name': actual_product_name,
                'main_category': self.current_main_category,
                'sub_category': self.current_sub_category,
                'options': ', '.join(options_list) if options_list else '',
                'option_price': option_price_value,
                'collect_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            self.save_product_info(product_info)
            print(f"\n상품 정보 수집 완료: {product_info['name']}")

        except Exception as e:
            print(f"상품 정보 수집 중 오류 발생: {str(e)}")
        finally:
            print("-" * 80)

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
            print("로그인 실패로 프로그램을 종료합니다.")

    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")
    
    finally:
        # 작업 완료 후 브라우저 종료
        automation.close()

if __name__ == "__main__":
    main()
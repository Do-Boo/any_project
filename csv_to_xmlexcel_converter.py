import pandas as pd
from datetime import datetime
import logging
import os
from xml.etree.ElementTree import Element, SubElement, parse, tostring
from xml.dom import minidom
import copy
import re

class CSVtoXMLExcelConverter:
    def __init__(self, csv_path: str = 'products.csv', xml_template_path: str = 'sample.xml'):
        self.csv_path = csv_path
        self.xml_template_path = xml_template_path
        self.logger = logging.getLogger(__name__)
        self.file_size_limit = 5 * 1024 * 1024  # 5MB
        
        # XML 네임스페이스 정의
        self.namespaces = {
            'ss': 'urn:schemas-microsoft-com:office:spreadsheet',
            'o': 'urn:schemas-microsoft-com:office:office',
            'x': 'urn:schemas-microsoft-com:office:excel',
            'html': 'http://www.w3.org/TR/REC-html40',
            'c': 'urn:schemas-microsoft-com:office:component:spreadsheet',
            'x2': 'http://schemas.microsoft.com/office/excel/2003/xml',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }

        # 필수 입력 항목 정의
        self.required_fields = [
            'category_code', 'product_name', 'sell_price', 'non_display'
        ]

        # 이미지 관련 필드
        self.image_fields = [
            'max_image', 'detail_image', 'tiny_image', 'mobile_image'
        ]

    def validate_file_size(self, file_path: str) -> bool:
        """파일 크기 검증"""
        if os.path.exists(file_path):
            return os.path.getsize(file_path) <= self.file_size_limit
        return True

    def validate_image_path(self, path: str) -> bool:
        """이미지 경로 검증"""
        if not path or path == 'AUTO':
            return True
                
        # 허용된 도메인 패턴
        allowed_domain = "doyou135.www267.freesell.co.kr/design/doyou135/product_images"
        
        # 도메인을 포함한 경로인 경우
        if allowed_domain in path:
            # 실제 이미지 파일명만 검사
            image_filename = path.split('/')[-1]
            
            # 허용되지 않는 문자 검사
            invalid_chars = set('!@#$%^&*()[]{}|\\;:\'",<>?~`')
            has_invalid_chars = any(char in image_filename for char in invalid_chars)
            
            # 한글 검사
            has_korean = bool(re.search('[가-힣]', image_filename))
            
            # 이미지 확장자 검사
            valid_extensions = ('.jpg', '.jpeg', '.gif')
            has_valid_extension = path.lower().endswith(valid_extensions)
            
            return not (has_invalid_chars or has_korean) and has_valid_extension
        
        # 상대 경로인 경우 (product_images로 시작하는 경우)
        elif path.startswith('product_images/'):
            image_filename = path.split('/')[-1]
            
            # 허용되지 않는 문자 검사
            invalid_chars = set('!@#$%^&*()[]{}|\\;:\'",<>?~`')
            has_invalid_chars = any(char in image_filename for char in invalid_chars)
            
            # 한글 검사
            has_korean = bool(re.search('[가-힣]', image_filename))
            
            # 이미지 확장자 검사
            valid_extensions = ('.jpg', '.jpeg', '.gif')
            has_valid_extension = path.lower().endswith(valid_extensions)
            
            return not (has_invalid_chars or has_korean) and has_valid_extension
        
        return False

    def validate_vat_type(self, vat_type: str) -> bool:
        """부가세 설정 검증"""
        valid_values = ['Y', 'N', 'Z', '면세상품', '부가세상품', '영세율상품']
        return str(vat_type).strip() in valid_values

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        try:
            # 필수 필드 검증
            if pd.isna(df.iloc[:, 0]).any():  # category_code
                raise ValueError("카테고리 코드는 필수 입력 항목입니다")
            if pd.isna(df.iloc[:, 1]).any():  # product_name
                raise ValueError("상품명은 필수 입력 항목입니다")
            if pd.isna(df.iloc[:, 11]).any():  # sell_price
                raise ValueError("판매가격은 필수 입력 항목입니다")

            processed_df = pd.DataFrame()
            
            # CSV 컬럼 매핑
            processed_df['category_code'] = df.iloc[:, 0].fillna('C000000')  # 분류미지정
            processed_df['product_uid'] = ""  # 상품 고유번호 (빈값)
            processed_df['provider_code'] = ""  # 공급자 코드
            processed_df['product_name'] = df.iloc[:, 1]  # 상품명
            processed_df['mobile_product_name'] = df.iloc[:, 2]  # 모바일 상품명
            
            # 옵션 관련 처리
            option_values = df.iloc[:, 6]  # 옵션값

            # 옵션 기본값 설정
            option_columns = ['opt_mandatory', 'opt_mix', 'opt_name', 'opt_value', 'opt_price',
                            'opt_image', 'opt_detail_image', 'opt_color', 'opt_type', 
                            'opt_use', 'opt_oneclick', 'opt_guide']
            for col in option_columns:
                processed_df[col] = ''

            # 옵션 데이터 처리
            for idx in df.index:
                opt_val = str(option_values[idx]).strip()
                if opt_val and opt_val != 'nan':
                    # 옵션값을 '|' 구분자로 분리 후 ',' 구분자로 재결합
                    options = [opt.strip() for opt in opt_val.split('|') if opt.strip()]
                    if options:
                        processed_df.at[idx, 'opt_mandatory'] = '필수'     
                        processed_df.at[idx, 'opt_mix'] = '미조합'         
                        processed_df.at[idx, 'opt_name'] = '기타'          
                        processed_df.at[idx, 'opt_value'] = ','.join(options)  # '|'를 ','로 변경
                        
                        # 모든 옵션의 가격을 0으로 설정
                        opt_prices = ['0'] * len(options)
                        processed_df.at[idx, 'opt_price'] = ','.join(opt_prices)
                        
                        processed_df.at[idx, 'opt_image'] = ''            
                        processed_df.at[idx, 'opt_detail_image'] = ''     
                        processed_df.at[idx, 'opt_color'] = ''            
                        processed_df.at[idx, 'opt_type'] = '선택형'        
                        processed_df.at[idx, 'opt_use'] = '사용'          
                        processed_df.at[idx, 'opt_oneclick'] = '사용'     
                        processed_df.at[idx, 'opt_guide'] = ''         

            # 재고 관련 필드
            stock_columns = {
                'opt_values': '',          
                'sto_type': '',           
                'sto_price': '',          
                'sto_stock': '',          
                'sto_unlimit': '',        
                'sto_order_stock': '',    
                'sto_safe_use': '',       
                'sto_safe_stock': '',     
                'sto_stop_use': '',       
                'sto_stop_stock': '',     
                'sto_code': '',           
                'sto_state': '',          
            }
            for col, val in stock_columns.items():
                processed_df[col] = val

            # 옵션 표시 타입 설정
            processed_df['option_display_type'] = ''
            for idx in df.index:
                opt_val = str(option_values[idx]).strip()
                if opt_val != '단일상품':
                    processed_df.at[idx, 'option_display_type'] = '일체형'

            # 가격 및 재고 정보
            processed_df['sell_price'] = df.iloc[:, 11]  # 판매가격
            processed_df['stock'] = df.iloc[:, 12]  # 재고수량
            processed_df['retail_price'] = df.iloc[:, 13]  # 소비자가격
            processed_df['original_price'] = df.iloc[:, 14]  # 구매원가
            processed_df['dicker'] = ""  # 가격대체문구
            processed_df['reserve'] = df.iloc[:, 15]  # 적립금
            processed_df['mobile_reserve'] = ""  # 모바일 적립금
            processed_df['point'] = ""  # 포인트

            # 기타 정보
            processed_df['key_word'] = df.iloc[:, 19]  # 키워드
            processed_df['sell_accept'] = "Y"  # 판매가능여부
            processed_df['non_display'] = '진열'  # 상품진열여부
            processed_df['max_image'] = df.iloc[:, 22].apply(
                lambda x: str(x).replace("product_images", "http://doyou135.www267.freesell.co.kr/design/doyou135/product_images") if pd.notna(x) and str(x).strip() else "AUTO"
            )
            processed_df['detail_image'] = 'AUTO'
            processed_df['tiny_image'] = 'AUTO'
            processed_df['mobile_image'] = df.iloc[:, 25].apply(
                lambda x: str(x).replace("product_images", "http://doyou135.www267.freesell.co.kr/design/doyou135/product_images") if pd.notna(x) else ""
            )
            processed_df['product_detail'] = df.iloc[:, 26].apply(
                lambda x: str(x).replace("product_images", "http://doyou135.www267.freesell.co.kr/design/doyou135/product_images") if pd.notna(x) else ""
            )
            processed_df['mobile_product_detail'] = df.iloc[:, 27].apply(
                lambda x: str(x).replace("product_images", "http://doyou135.www267.freesell.co.kr/design/doyou135/product_images") if pd.notna(x) else ""
            )
            processed_df['model_name'] = df.iloc[:, 28]
            # processed_df['vat_type'] = 'Y'  # 부가세 설정

            # 이미지 경로 검증
            for field in self.image_fields:
                if field in processed_df.columns:
                    invalid_paths = processed_df[field].apply(
                        lambda x: not self.validate_image_path(str(x)) if pd.notna(x) else False
                    )
                    if invalid_paths.any():
                        raise ValueError(f"이미지 경로에 한글/공백/특수기호가 포함되어 있습니다: {field}")

            return processed_df
                
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류: {str(e)}")
            raise

    def create_cell(self, value, style_id="s65"):
        """스타일을 유지하면서 셀 생성"""
        cell = Element('{%s}Cell' % self.namespaces['ss'])
        cell.set('{%s}StyleID' % self.namespaces['ss'], style_id)
        
        if pd.notna(value) and str(value).strip():
            data = SubElement(cell, '{%s}Data' % self.namespaces['ss'])
            data.set('{%s}Type' % self.namespaces['ss'], 'String')
            data.text = str(value)
        
        return cell

    def convert(self) -> bool:
        try:
            # CSV 읽기 전 파일 존재 확인
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.csv_path}")
            
            # XML 템플릿 존재 확인
            if not os.path.exists(self.xml_template_path):
                raise FileNotFoundError(f"XML 템플릿 파일을 찾을 수 없습니다: {self.xml_template_path}")
            
            # CSV 읽기
            df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            if df.empty:
                raise Exception("CSV 데이터가 비어있습니다.")
            
            # 데이터 전처리
            processed_df = self.prepare_data(df)
            
            # XML 템플릿 파싱
            tree = parse(self.xml_template_path)
            root = tree.getroot()
            
            # Worksheet 찾기
            worksheet = root.find('.//{%s}Worksheet' % self.namespaces['ss'])
            if worksheet is None:
                raise Exception("XML 템플릿에서 Worksheet를 찾을 수 없습니다.")
            
            table = worksheet.find('.//{%s}Table' % self.namespaces['ss'])
            if table is None:
                raise Exception("XML 템플릿에서 Table을 찾을 수 없습니다.")
            
            # 기존의 빈 데이터 행들 제거 (3번째 행부터)
            template_row = None
            rows = table.findall('.//{%s}Row' % self.namespaces['ss'])
            for i, row in enumerate(rows):
                if i < 2:  # 헤더 2줄은 유지
                    continue
                elif i == 2:  # 3번째 행의 스타일을 템플릿으로 저장
                    template_row = copy.deepcopy(row)
                table.remove(row)
            
            # 새로운 데이터 행 추가
            for _, row_data in processed_df.iterrows():
                new_row = Element('{%s}Row' % self.namespaces['ss'])
                new_row.set('{%s}AutoFitHeight' % self.namespaces['ss'], '0')
                new_row.set('{%s}Height' % self.namespaces['ss'], '12.75')
                
                # 각 필드에 대해 셀 생성
                for column in processed_df.columns:
                    cell = self.create_cell(row_data[column])
                    new_row.append(cell)
                
                table.append(new_row)
            
            # 임시 파일로 저장
            output_file = f'products_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xml'
            temp_file = f'temp_{output_file}'
            
            # XML 문자열 생성 시 선언과 인코딩 정보 추가
            xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n'
            xml_str += '<?mso-application progid="Excel.Sheet"?>\n'
            xml_str += tostring(root, encoding='unicode')
            
            # 임시 파일로 저장
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            # 파일 크기 검증
            if not self.validate_file_size(temp_file):
                os.remove(temp_file)
                raise Exception("생성된 파일이 5MB 크기 제한을 초과합니다.")
            
            # 임시 파일을 최종 파일로 이동
            os.rename(temp_file, output_file)
            
            self.logger.info(f"XML Excel 파일 생성 완료: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"XML Excel 변환 실패: {str(e)}")
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
            return False

def main():
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 콘솔 출력
            logging.FileHandler('converter.log', encoding='utf-8')  # 파일 출력
        ]
    )
    
    try:
        # 변환기 인스턴스 생성
        converter = CSVtoXMLExcelConverter()
        
        # 변환 실행
        if converter.convert():
            print("CSV 파일이 성공적으로 XML Excel로 변환되었습니다.")
        else:
            print("변환 중 오류가 발생했습니다. 로그 파일을 확인해주세요.")
            
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
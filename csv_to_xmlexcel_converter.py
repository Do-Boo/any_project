import pandas as pd
from datetime import datetime
import logging
import os
from xml.etree.ElementTree import Element, SubElement, parse, tostring
from xml.dom import minidom
import copy

class CSVtoXMLExcelConverter:
    def __init__(self, csv_path: str = 'products.csv', xml_template_path: str = 'sample.xml'):
        self.csv_path = csv_path
        self.xml_template_path = xml_template_path
        self.logger = logging.getLogger(__name__)
        
        # XML 네임스페이스 정의
        self.namespaces = {
            'ss': 'urn:schemas-microsoft-com:office:spreadsheet',
            'o': 'urn:schemas-microsoft-com:office:office',
            'x': 'urn:schemas-microsoft-com:office:excel',
            'html': 'http://www.w3.org/TR/REC-html40'
        }

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        try:
            processed_df = pd.DataFrame()
            
            # CSV 컬럼 매핑 (CSV 파일의 실제 구조에 맞춤)
            processed_df['category_code'] = df.iloc[:, 0]  # 카테고리 코드
            processed_df['product_uid'] = ""  # 상품 고유번호 (빈값)
            processed_df['provider_code'] = ""  # 공급자 코드
            processed_df['product_name'] = df.iloc[:, 1]  # 상품명
            processed_df['mobile_product_name'] = df.iloc[:, 2]  # 모바일 상품명
            
            # 옵션 관련 처리
            option_values = df.iloc[:, 10]  # 옵션값

            # 옵션 기본값 설정
            option_columns = ['opt_mandatory', 'opt_mix', 'opt_name', 'opt_value', 'opt_price',
                            'opt_image', 'opt_detail_image', 'opt_color', 'opt_type', 
                            'opt_use', 'opt_oneclick', 'opt_guide']
            for col in option_columns:
                processed_df[col] = ''

            for idx in df.index:
                opt_val = str(option_values[idx]).strip()
                if opt_val and opt_val != '단일상품':
                    options = [opt.strip() for opt in opt_val.split('|') if opt.strip()]
                    if options:
                        processed_df.at[idx, 'opt_mandatory'] = '필수'     # F열
                        processed_df.at[idx, 'opt_mix'] = '미조합'         # G열
                        processed_df.at[idx, 'opt_name'] = '기타'          # H열
                        processed_df.at[idx, 'opt_value'] = ','.join(options)  # I열
                        processed_df.at[idx, 'opt_price'] = ','.join(['0'] * len(options))  # J열
                        processed_df.at[idx, 'opt_image'] = ''            # K열
                        processed_df.at[idx, 'opt_detail_image'] = ''     # L열
                        processed_df.at[idx, 'opt_color'] = ''            # M열
                        processed_df.at[idx, 'opt_type'] = '선택형'        # N열
                        processed_df.at[idx, 'opt_use'] = '사용'          # O열
                        processed_df.at[idx, 'opt_oneclick'] = '사용'     # P열
                        processed_df.at[idx, 'opt_guide'] = ''           # Q열

            # 재고 관련 필드 (R~AD열)
            stock_columns = {
                'opt_values': '',          # R열
                'sto_type': '',           # S열
                'sto_price': '',          # T열
                'sto_stock': '',          # U열
                'sto_unlimit': '',        # V열
                'sto_order_stock': '',    # W열
                'sto_safe_use': '',       # X열
                'sto_safe_stock': '',     # Y열
                'sto_stop_use': '',       # Z열
                'sto_stop_stock': '',     # AA열
                'sto_code': '',           # AB열
                'sto_state': '',          # AC열
            }
            for col, val in stock_columns.items():
                processed_df[col] = val

            # 가격 및 재고 정보
            processed_df['option_display_type'] = ''  # 기본값 설정
            for idx in df.index:
                opt_val = str(option_values[idx]).strip()
                if opt_val != '단일상품':
                    processed_df.at[idx, 'option_display_type'] = '일체형'
                else:
                    processed_df.at[idx, 'option_display_type'] = ''

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
                lambda x: str(x).replace("product_images", "doyou135.www267.freesell.co.kr/design/doyou135/product_images") if pd.notna(x) else ""
            )  # 확대이미지
            processed_df['detail_image'] = 'AUTO'  # 상세이미지
            processed_df['tiny_image'] = 'AUTO'  # 리스트이미지
            processed_df['mobile_image'] = df.iloc[:, 25].apply(
                lambda x: str(x).replace("product_images", "doyou135.www267.freesell.co.kr/design/doyou135/product_images") if pd.notna(x) else ""
            )  # 모바일 이미지
            processed_df['product_detail'] = df.iloc[:, 26]  # 상품상세
            processed_df['mobile_product_detail'] = df.iloc[:, 27]  # 모바일 상세설명
            processed_df['model_name'] = df.iloc[:, 28]  # 모델명
            # processed_df['best_product_display'] = 'N'  # 베스트상품 진열여부
            # processed_df['vat_type'] = 'Y'  # 부가세 설정

            return processed_df
                
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류: {str(e)}")
            raise

    def create_cell(self, value, style_id="s65"):
        """스타일을 유지하면서 셀 생성"""
        cell = Element('{%s}Cell' % self.namespaces['ss'])
        cell.set('{%s}StyleID' % self.namespaces['ss'], style_id)
        
        data = SubElement(cell, '{%s}Data' % self.namespaces['ss'])
        data.set('{%s}Type' % self.namespaces['ss'], 'String')
        data.text = str(value) if pd.notna(value) else ""
        
        return cell

    def convert(self):
        try:
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
            table = worksheet.find('.//{%s}Table' % self.namespaces['ss'])
            
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
            
            # 파일 저장
            output_file = f'products_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xml'
            
            # XML 문자열 생성 시 선언과 인코딩 정보 추가
            xml_str = '<?xml version="1.0"?>\n<?mso-application progid="Excel.Sheet"?>\n'
            xml_str += tostring(root, encoding='unicode')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            self.logger.info(f"XML Excel 파일 생성 완료: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"XML Excel 변환 실패: {str(e)}")
            return False

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('converter.log')
        ]
    )
    
    converter = CSVtoXMLExcelConverter()
    if converter.convert():
        print("CSV 파일이 성공적으로 XML Excel로 변환되었습니다.")
    else:
        print("변환 중 오류가 발생했습니다. 로그 파일을 확인해주세요.")

if __name__ == "__main__":
    main()
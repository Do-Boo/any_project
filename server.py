import asyncio
from playwright.async_api import async_playwright
import websockets
import json
import time

class WebCapture:
    def __init__(self, relay_server_url):
        self.relay_url = relay_server_url
        self.playwright = None
        self.browser = None
        self.page = None
        self.last_update_time = 0
        self.focused_element = None
        self.last_input_value = None
        
    async def init_browser(self):
        """브라우저 초기화"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page()

    async def handle_client_event(self, event_data):
        """클라이언트 이벤트 처리"""
        try:
            event_type = event_data['type']
            
            if event_type == 'mouse':
                # 마우스 이벤트 처리
                x = event_data.get('x', 0)
                y = event_data.get('y', 0)
                
                # 마우스 이동
                await self.page.mouse.move(x, y)
                
                # 클릭 이벤트가 있는 경우
                if event_data.get('click'):
                    button = event_data.get('button', 'left')  # 기본값은 왼쪽 클릭
                    await self.page.mouse.click(x, y, button=button)
                    print(f"마우스 클릭: x={x}, y={y}, button={button}")
                else:
                    print(f"마우스 이동: x={x}, y={y}")
                    
            elif event_type == 'keyboard':
                # 키보드 이벤트 처리
                if event_data.get('key'):
                    key = event_data['key']
                    await self.page.keyboard.press(key)
                    print(f"키 입력: {key}")
                    
                if event_data.get('text'):
                    text = event_data['text']
                    await self.page.keyboard.type(text)
                    print(f"텍스트 입력: {text}")
                    
            elif event_type == 'scroll':
                # 스크롤 이벤트 처리
                x = event_data.get('x', 0)
                y = event_data.get('y', 0)
                await self.page.evaluate(f'window.scrollTo({x}, {y})')
                print(f"스크롤: x={x}, y={y}")
                
        except Exception as e:
            print(f"이벤트 처리 중 에러: {e}")

    async def get_dom_changes(self):
        """DOM 변경사항 감지 및 수집"""
        return await self.page.evaluate('''() => {
            return new Promise((resolve) => {
                // 현재 활성화된 요소와 입력 상태 저장
                let activeElement = document.activeElement;
                let focusState = null;

                if (activeElement && 
                    (activeElement.tagName === 'INPUT' || 
                     activeElement.tagName === 'TEXTAREA' || 
                     activeElement.contentEditable === 'true')) {
                    
                    // 활성화된 요소의 DOM 경로 계산
                    let path = [];
                    let node = activeElement;
                    while (node && node !== document.documentElement) {
                        let parent = node.parentNode;
                        if (!parent) break;
                        
                        let index = Array.from(parent.children).indexOf(node);
                        path.unshift(index);
                        node = parent;
                    }

                    // 포커스 상태 정보 저장
                    focusState = {
                        path: path,
                        value: activeElement.value || activeElement.textContent,
                        selectionStart: activeElement.selectionStart,
                        selectionEnd: activeElement.selectionEnd
                    };
                }

                // 변경사항을 감지하는 MutationObserver 설정
                let changes = [];
                const observer = new MutationObserver((mutations) => {
                    mutations.forEach(mutation => {
                        // 입력 중인 요소의 변경은 무시
                        if (activeElement && 
                            (mutation.target === activeElement || 
                             activeElement.contains(mutation.target))) {
                            return;
                        }

                        // 변경사항 수집
                        let path = [];
                        let node = mutation.target;
                        while (node && node !== document.documentElement) {
                            let parent = node.parentNode;
                            if (!parent) break;
                            
                            let index = Array.from(parent.children).indexOf(node);
                            path.unshift(index);
                            node = parent;
                        }

                        changes.push({
                            type: mutation.type,
                            path: path,
                            data: {
                                outerHTML: mutation.target.outerHTML,
                                attributeName: mutation.attributeName,
                                attributeValue: mutation.target.getAttribute(mutation.attributeName),
                                oldValue: mutation.oldValue,
                                addedNodes: Array.from(mutation.addedNodes).map(n => n.outerHTML || ''),
                                removedNodes: Array.from(mutation.removedNodes).map(n => n.outerHTML || '')
                            }
                        });
                    });

                    if (changes.length > 0) {
                        resolve({
                            type: 'update',
                            changes: changes,
                            focusState: focusState,
                            timestamp: Date.now()
                        });
                    }
                });

                // 모든 DOM 변경 감지
                observer.observe(document.documentElement, {
                    childList: true,
                    attributes: true,
                    characterData: true,
                    subtree: true,
                    attributeOldValue: true,
                    characterDataOldValue: true
                });

                // 초기 상태 전송
                resolve({
                    type: 'initial',
                    html: document.documentElement.outerHTML,
                    styles: Array.from(document.styleSheets).map(sheet => {
                        try {
                            return Array.from(sheet.cssRules).map(rule => rule.cssText);
                        } catch (e) {
                            return [];
                        }
                    }).flat(),
                    focusState: focusState,
                    timestamp: Date.now()
                });
            });
        }''')

    async def connect_websocket(self):
        """WebSocket 연결 시도"""
        for retry in range(3):
            try:
                return await websockets.connect(
                    self.relay_url,
                    ping_interval=20,
                    ping_timeout=20
                )
            except Exception as e:
                print(f"연결 시도 {retry + 1} 실패: {e}")
                if retry < 2:
                    await asyncio.sleep(5)
        return None

    async def capture_and_send(self):
        """웹 페이지의 DOM 변경사항을 캡처하고 전송"""
        while True:
            try:
                websocket = await self.connect_websocket()
                if not websocket:
                    print("WebSocket 연결 실패")
                    await asyncio.sleep(5)
                    continue

                print("WebSocket 연결 성공!")
                
                # 회사 PC 인증
                await websocket.send(json.dumps({
                    'type': 'company_auth',
                    'token': 'company-secret-token'
                }))
                print("인증 메시지 전송 완료")

                # 초기 상태 전송
                initial_data = await self.get_dom_changes()
                await websocket.send(json.dumps({
                    'type': 'dom_update',
                    'data': initial_data,
                    'timestamp': time.time()
                }))
                print("초기 DOM 데이터 전송 완료")

                while True:
                    try:
                        # 클라이언트로부터의 이벤트 수신
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        # 이벤트 처리
                        if data.get('type') in ['mouse', 'keyboard', 'scroll']:
                            await self.handle_client_event(data)
                        
                        # 주기적인 DOM 업데이트 확인 및 전송
                        current_time = time.time()
                        if current_time - self.last_update_time >= 1.0:
                            dom_data = await self.get_dom_changes()
                            
                            if (dom_data.get('type') == 'update' and 
                                len(dom_data.get('changes', [])) > 0) or \
                               dom_data.get('type') == 'initial':
                                
                                await websocket.send(json.dumps({
                                    'type': 'dom_update',
                                    'data': dom_data,
                                    'timestamp': current_time
                                }))
                                print(f"DOM 업데이트 전송: {len(dom_data.get('changes', []))}개 변경사항")
                            
                            self.last_update_time = current_time
                        
                        await asyncio.sleep(0.1)  # CPU 사용량 조절
                        
                    except Exception as e:
                        print(f"데이터 처리 중 에러: {e}")
                        break
                        
            except Exception as e:
                print(f"WebSocket 연결 에러: {e}")
                if websocket:
                    await websocket.close()
                await asyncio.sleep(5)
                    
    async def start(self):
        """캡처 시작"""
        await self.init_browser()
        
        # 특정 URL로 이동
        await self.page.goto('https://www.naver.com')
        print("웹 페이지 로딩 완료")
        
        # WebSocket 연결 및 데이터 전송
        await self.capture_and_send()
        
    async def stop(self):
        """리소스 정리"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

async def main():
    capturer = WebCapture('ws://doboo.tplinkdns.com:8088')
    try:
        await capturer.start()
    except KeyboardInterrupt:
        await capturer.stop()
    except Exception as e:
        print(f"예상치 못한 에러 발생: {e}")
        await capturer.stop()

if __name__ == "__main__":
    asyncio.run(main())
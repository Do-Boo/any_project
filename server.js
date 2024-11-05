const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// 클라이언트와 회사 PC 관리
const clients = new Set();
let companyPC = null;
let lastDomData = null;

// 메인 페이지 제공
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// WebSocket 연결 관리
wss.on('connection', (ws) => {
    console.log('새로운 연결 수립');
    clients.add(ws);

    // 연결 초기화
    ws.isAlive = true;
    ws.on('pong', () => {
        ws.isAlive = true;
    });

    // 메시지 처리
    ws.on('message', async (message) => {
        try {
            const data = JSON.parse(message);

            // 회사 PC 인증 처리
            if (data.type === 'company_auth') {
                ws.isCompanyPC = true;
                companyPC = ws;
                console.log('회사 PC 인증 완료');
                return;
            }

            // DOM 업데이트 처리 (회사 PC -> 클라이언트)
            if (data.type === 'dom_update' && ws.isCompanyPC) {
                lastDomData = data;

                // 모든 클라이언트에게 전송
                clients.forEach(client => {
                    if (client !== ws && client.readyState === WebSocket.OPEN) {
                        client.send(JSON.stringify(data));
                    }
                });

                // 업데이트 타입에 따른 로깅
                if (data.data.type === 'initial') {
                    console.log('초기 DOM 데이터 전송됨');
                } else {
                    console.log(`DOM 업데이트: ${data.data.changes.length}개 변경사항`);
                }
                return;
            }

            // 클라이언트 이벤트 처리 (클라이언트 -> 회사 PC)
            if (['mouse', 'keyboard', 'scroll'].includes(data.type)) {
                if (companyPC && companyPC.readyState === WebSocket.OPEN) {
                    companyPC.send(JSON.stringify(data));
                    console.log(`${data.type} 이벤트 전달됨`);
                }
                return;
            }

        } catch (error) {
            console.error('메시지 처리 중 에러:', error);
        }
    });

    // 연결 종료 처리
    ws.on('close', () => {
        clients.delete(ws);
        if (ws.isCompanyPC) {
            companyPC = null;
            console.log('회사 PC 연결 종료');
        }
        console.log(`연결 종료. 현재 클라이언트 수: ${clients.size}`);
    });

    // 에러 처리
    ws.on('error', (error) => {
        console.error('WebSocket 에러:', error);
        clients.delete(ws);
        if (ws.isCompanyPC) {
            companyPC = null;
        }
    });

    // 신규 클라이언트에게 마지막 DOM 상태 전송
    if (lastDomData && !ws.isCompanyPC) {
        ws.send(JSON.stringify(lastDomData));
    }
});

// 연결 상태 확인
const interval = setInterval(() => {
    wss.clients.forEach(ws => {
        if (!ws.isAlive) {
            clients.delete(ws);
            if (ws.isCompanyPC) {
                companyPC = null;
            }
            return ws.terminate();
        }

        ws.isAlive = false;
        ws.ping();
    });
}, 30000);

// 서버 종료 시 정리
wss.on('close', () => {
    clearInterval(interval);
});

// 서버 시작
const PORT = process.env.PORT || 8088;
server.listen(PORT, () => {
    console.log(`서버가 포트 ${PORT}에서 실행 중입니다`);
});

// 종료 처리
process.on('SIGTERM', () => {
    console.log('서버를 종료합니다...');
    server.close(() => {
        console.log('서버가 종료되었습니다');
        process.exit(0);
    });
});

process.on('SIGINT', () => {
    console.log('서버를 종료합니다...');
    server.close(() => {
        console.log('서버가 종료되었습니다');
        process.exit(0);
    });
});
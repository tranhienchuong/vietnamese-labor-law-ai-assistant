import { NextRequest } from "next/server"

export const runtime = "nodejs"

type ChatMessage = {
  role: "system" | "user" | "assistant"
  content: string
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function chunkText(text: string, size = 24) {
  const chunks: string[] = []
  for (let index = 0; index < text.length; index += size) {
    chunks.push(text.slice(index, index + size))
  }
  return chunks
}

function normalizeVietnamese(text: string) {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
}

function buildDemoAnswer(question: string) {
  const lowerQuestion = normalizeVietnamese(question)

  if (lowerQuestion.includes("boi thuong") || lowerQuestion.includes("trai luat")) {
    return `Theo Bộ luật Lao động 2019, nếu người sử dụng lao động đơn phương chấm dứt hợp đồng lao động trái pháp luật, hướng xử lý thường gồm:

1. Nhận người lao động trở lại làm việc theo hợp đồng đã giao kết.
2. Trả tiền lương, đóng bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp cho thời gian người lao động không được làm việc.
3. Bồi thường thêm ít nhất 02 tháng tiền lương theo hợp đồng lao động.
4. Nếu người lao động không muốn quay lại hoặc doanh nghiệp không còn vị trí phù hợp, cần xem thêm khoản bồi thường/thỏa thuận tương ứng.

Căn cứ pháp lý nên kiểm tra: Điều 41 Bộ luật Lao động 2019.

Lưu ý: cần đối chiếu lý do chấm dứt, trình tự báo trước, biên bản và hồ sơ nhân sự trước khi kết luận cuối cùng.`
  }

  if (lowerQuestion.includes("tro cap") || lowerQuestion.includes("thoi viec")) {
    return `Về nguyên tắc, trợ cấp thôi việc theo Bộ luật Lao động 2019 được xem xét khi người lao động đã làm việc thường xuyên từ đủ 12 tháng trở lên và hợp đồng chấm dứt thuộc trường hợp luật định.

Cách tiếp cận an toàn:

- Xác định tổng thời gian làm việc thực tế.
- Trừ thời gian đã tham gia bảo hiểm thất nghiệp và thời gian đã được chi trả trợ cấp.
- Mức trợ cấp thường tính theo mỗi năm làm việc bằng 1/2 tháng tiền lương, nhưng cần kiểm tra dữ kiện cụ thể.

Căn cứ pháp lý nên kiểm tra: Điều 46 Bộ luật Lao động 2019.`
  }

  return `Với câu hỏi này, cần xác định trước loại hợp đồng, chủ thể muốn chấm dứt và lý do chấm dứt.

Nếu người lao động đơn phương chấm dứt hợp đồng:

- Hợp đồng không xác định thời hạn: thường phải báo trước ít nhất 45 ngày.
- Hợp đồng xác định thời hạn từ 12 đến 36 tháng: thường phải báo trước ít nhất 30 ngày.
- Hợp đồng dưới 12 tháng: thường phải báo trước ít nhất 03 ngày làm việc.
- Một số trường hợp người lao động có thể nghỉ mà không cần báo trước, ví dụ không được bố trí đúng công việc, không được trả đủ lương hoặc bị quấy rối tại nơi làm việc.

Căn cứ pháp lý nên kiểm tra: Điều 35 Bộ luật Lao động 2019.`
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}))
  const backendUrl = process.env.BACKEND_URL?.trim()

  if (backendUrl) {
    const response = await fetch(`${backendUrl.replace(/\/$/, "")}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body)
    })

    if (!response.ok || !response.body) {
      return new Response("Backend error", { status: 502 })
    }

    return new Response(response.body, {
      headers: {
        "Cache-Control": "no-cache",
        "Content-Type":
          response.headers.get("Content-Type") ?? "text/plain; charset=utf-8"
      }
    })
  }

  const messages = Array.isArray(body.messages)
    ? (body.messages as ChatMessage[])
    : []
  const lastUserMessage =
    [...messages].reverse().find((message) => message.role === "user")?.content ??
    ""
  const answer = buildDemoAnswer(lastUserMessage)
  const encoder = new TextEncoder()

  const stream = new ReadableStream({
    async start(controller) {
      for (const chunk of chunkText(answer)) {
        controller.enqueue(encoder.encode(chunk))
        await sleep(26)
      }
      controller.close()
    }
  })

  return new Response(stream, {
    headers: {
      "Cache-Control": "no-cache",
      "Content-Type": "text/plain; charset=utf-8"
    }
  })
}

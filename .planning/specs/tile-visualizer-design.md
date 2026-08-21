# Thiết kế: Công cụ mô phỏng vật liệu cho showroom

Trạng thái: **bản thảo, chờ anh review** · Ngày 2026-08-20
Bằng chứng kỹ thuật: `.planning/specs/tile-render-spike/README.md`

---

## 1. Bối cảnh

Showroom trang trí nội thất, bán **gạch men + thiết bị vệ sinh**.

Cơ cấu doanh thu (theo ước lượng của chủ, chưa có số liệu):

| Nhóm khách | Tỉ trọng | Có cần mô phỏng? |
|---|---|---|
| Chủ nhà tự xây/sửa | **>50%** | Có, rất cần |
| Nhà thầu | phần còn lại | Không — cần báo giá nhanh, đúng lô |
| Đại lý bán lại | phần còn lại | Không — cần giá sỉ, tồn kho |

Khách chủ nhà **mua cho cả nhà cùng một lần**, không mua từng phòng.

Bốn điểm nghẽn được xác định trong hành trình tư vấn: khách không hình dung được;
báo giá chậm/sai; khách rời showroom là mất dấu; khách choáng vì quá nhiều mẫu.

**Nhận định trung tâm:** cả bốn đều đọc từ *một* cấu trúc dữ liệu — hồ sơ không gian
của khách (ảnh + kích thước + mẫu đã chọn). Vì vậy dựng xương sống trước, treo từng
tính năng lên sau, chứ không làm bốn tính năng rời rạc.

---

## 2. Phạm vi v1

**Trong phạm vi**

- Gạch **sàn**
- Hai đường render, chọn theo loại phòng (mục 4)
- Loại trừ vật che bằng **vẽ tay**, cộng tự động theo màu như tiện ích
- Hồ sơ dự án nhiều phòng
- Tính khối lượng và báo giá từ cùng bộ kích thước
- Link hồ sơ khách tự copy để mang về

**Ngoài phạm vi v1**

- Gạch **ốp tường** — nhiều mặt phẳng, bị che nhiều, bài toán khác hẳn
- **Thiết bị vệ sinh** — không phải dán texture mà là đặt vật thể 3D, cần model từng SP
- Tích hợp Zalo OA — bỏ có chủ ý; link thủ công đạt cùng kết quả, không cần xác minh doanh nghiệp
- Segmentation bằng model AI — xem mục 5, đã chứng minh không cần cho v1
- Gợi ý/lọc mẫu tự động (điểm nghẽn "choáng vì nhiều mẫu") — giai đoạn sau

---

## 3. Mô hình dữ liệu

```
Dự án (một khách, một lần mua)
├── Phòng  (N phòng)
│   ├── loại phòng        → quyết định đường render
│   ├── kích thước (m)    → BẮT BUỘC, dùng cho cả render và báo giá
│   ├── nguồn ảnh         → ảnh khách chụp (đường B) | cảnh mẫu (đường A)
│   ├── tứ giác mặt sàn   → CHỈ đường B: 4 điểm chấm tay
│   ├── vùng loại trừ     → CHỈ đường B: đa giác vẽ tay quanh vật che
│   └── SKU đã chọn + kiểu lát + màu/độ rộng ke
└── Báo giá  → gộp NGANG QUA các phòng, theo SKU
```

**Kích thước phải do người nhập, hoặc suy từ vật mốc đã biết trong ảnh.** Ước lượng
độ sâu từ một ảnh chỉ cho tỉ lệ tương đối, **không** cho kích thước thật. Đây là ràng
buộc vật lý, không phải thiếu sót triển khai.

Cảnh mẫu (đường A) đã có mặt phẳng sàn định sẵn khi dựng, nên hai trường `tứ giác` và
`vùng loại trừ` để trống — khách không phải chấm gì. Nhưng **kích thước vẫn bắt buộc**,
vì báo giá cần nó dù render không cần.

Một lần nhập kích thước → hai đầu ra: render đúng tỉ lệ, và báo giá.

---

## 4. Hai đường render, chọn theo loại phòng

| Loại phòng | Đường | Vì sao |
|---|---|---|
| Phòng khách, phòng ngủ | **B — ảnh thật của khách** | Sàn rộng liền mạch, vật che thường có màu rõ, và khách *cần* thấy nhà mình |
| WC, bếp | **A — cảnh mẫu dựng sẵn** | Sàn nhỏ vụn, vật che **trắng/trung tính** (masking màu chết ở đây), và ảnh nhà thật ít giá trị vì WC nhìn khá giống nhau |

Đo trên một nhà phố 7 phòng thực tế: **72% diện tích** đi đường B, **28%** đi đường A.

Cách chia này làm **rủi ro che khuất tự tan bằng phạm vi**, không phải giải bằng AI.
Đường A cũng đóng vai đường lùi khi ảnh khách quá tệ để dùng.

### Đường B — bốn bước, không có AI

1. **Mặt sàn** — 4 điểm chấm tay, trừ các đa giác loại trừ
2. **Homography** — `getPerspectiveTransform`, ma trận 3×3 từ 4 điểm
3. **Trải texture** — đúng tỉ lệ vật lý, có ke, xen 4–8 mặt vân với seed cố định
4. **Ghép sáng** — nhân độ sáng tần số thấp của ảnh gốc lên texture mới

Bước 4 là bước làm ảnh trông thật. Thiếu nó thì sàn mới như dán sticker.

**Đã xác nhận chạy được.** Chi tiết và hai bug đã sửa: xem README của spike.

---

## 5. Chiến lược loại trừ vật che

**Vẽ tay là chính. Tự động là tiện ích.** Không phải sở thích — đây là kết luận đo được.

Thử một lớp mô hình màu (học màu sàn từ dải sát mép gần, đánh dấu pixel lệch):

| Ngưỡng | Giữ được tủ gỗ | Giữ được vật trung tính | Sàn thật còn nguyên |
|---|---|---|---|
| 13 | 100% | 23% | 100% |
| 6 | 100% | 100% | **37%** |
| 3 | 100% | 100% | **0%** |

**Không tồn tại ngưỡng nào vừa giữ được vật trung tính vừa giữ được sàn.** Ngưỡng
càng thấp để bắt vật xám thì càng ăn vào sàn thật.

Hệ quả trực tiếp cho ngành hàng này: **thiết bị vệ sinh trắng trên gạch sáng là đúng
ca không thể tách bằng màu.** Vì vậy mục 4 định tuyến WC sang đường A.

Ràng buộc giao diện: vẽ tay phải là **bút vẽ tự do**, không phải kéo hộp. Hộp loại trừ
quá tay và để lại vệt sàn chưa lát quanh mỗi vật.

GrabCut đã thử và **sai về nguyên lý**: mồi nền của nó là vùng ngoài tứ giác, tức là
tường — cùng màu xám với sàn. Hai mô hình màu trùng nhau nên không phân biệt được gì.

---

## 6. Báo giá

Gộp theo SKU **ngang qua các phòng** rồi mới làm tròn thùng, không làm tròn từng phòng.

Đo trên nhà phố 7 phòng / 63,74 m² / 4 SKU: 50 thùng → 48 thùng. **Tiết kiệm 4%.**

**Đừng dựng câu chuyện bán hàng trên 4% này.** Giá trị thật là **đồng lô**: đặt 23
thùng thành *một* dòng hàng thì kho lấy cùng một lô. Ba dòng riêng cho ba phòng ngủ có
thể được lấy từ ba lô sản xuất, và ba phòng "cùng mã gạch" ra ba tông màu. Đó mới là
lỗi đắt.

Báo giá cần các trường: kích thước viên, số viên/thùng, m²/thùng, giá bán, hao hụt
(mặc định 8%). Thiếu `m²/thùng` là không tự động hoá được.

Con số 8% là **giả định cần anh xác nhận**: hao hụt thật đổi theo kích thước viên và
kiểu lát — lát chéo hao nhiều hơn lát thẳng đáng kể. Nên để nó là tham số theo SKU và
theo kiểu lát, đừng chôn cứng một số.

---

## 7. Dữ liệu sản phẩm — tảng băng

Ba trong bốn tính năng chết nếu không có dữ liệu số hoá. Mỗi SKU cần:

- **Texture lặp liền mạch**, và **đủ 4–8 mặt vân**. Thiếu mặt vân thì sàn render ra lặp
  y một viên, khách nhìn là biết giả
- Kích thước viên, bề mặt (bóng/mờ/nhám)
- Giá, số viên/thùng, m²/thùng

**Ảnh catalogue của hãng thường không dùng được** — đó là ảnh chụp viên gạch có bóng
đổ, có góc nghiêng. Cần file chụp thẳng mặt, không bóng.

**Trạng thái: CHƯA BIẾT.** Đang chờ khảo sát nội bộ và hỏi nhà cung cấp. Đây là ẩn số
lớn nhất còn lại về tiến độ, và nó quyết định dự án này là vài tuần hay vài tháng.

Chiến lược giảm rủi ro: **không làm cả catalogue — làm 30–60 mẫu bán chạy nhất.**

---

## 8. Kỹ thuật

| Lớp | Chọn | Lý do |
|---|---|---|
| Giao diện | Web app / PWA, Next.js + TypeScript | Sales dùng tablet, khách dùng điện thoại; không bắt khách tải app |
| Đổi gạch | WebGL tại máy khách | Khách lật 20 mẫu liên tiếp — phải thấy liền, không chờ server |
| Phân tích ảnh | Python + OpenCV (FastAPI), **1 lần/ảnh** | Mask + homography tính một lần rồi lưu |
| Dữ liệu | PostgreSQL | Dự án, phòng, SKU, báo giá |
| Ảnh & texture | Object storage | Texture nhiều và nặng |
| Hạ tầng | VPS thường, **không GPU** | Đường hình học không cần GPU; lưu lượng showroom rất nhỏ |

Không có tích hợp bên thứ ba nào trong v1.

**Quyết định kiến trúc đáng chú ý:** tách hai pha — phân tích ảnh nặng làm một lần trên
server, đổi texture làm tức thì tại máy khách. Nếu mỗi lần đổi mẫu phải chờ server thì
công cụ chết vì khách mất kiên nhẫn, dù ảnh đẹp cỡ nào.

---

## 9. Rủi ro còn mở

1. **Độ thật — chưa xác nhận.** Toàn bộ thiết kế đứng hay sụp ở câu "khách có tin đó là
   sàn nhà mình không". Ảnh test tới giờ đều là ảnh tự dựng, không trả lời được câu đó.
   **Cần: 1–2 ảnh phòng thật chụp bằng điện thoại + 1–2 ảnh texture kèm kích thước mm.**
2. **Dữ liệu sản phẩm — chưa biết** (mục 7). Ẩn số tiến độ lớn nhất.
3. **Gạch bóng chưa có phản chiếu.** Gạch mờ ổn; gạch bóng sẽ trông phẳng.
4. **Răng cưa mép xa.** Gạch nén mạnh ở xa, bản thật cần mipmap.
5. **Tỉ trọng nhóm khách là ước lượng**, chưa đối chiếu số liệu bán hàng.
6. **Khách đang ở giai đoạn nào chưa rõ** — nhà thô hay nhà hoàn thiện. Đường A che
   được rủi ro này, nhưng nếu phần lớn khách là nhà thô thì đường B mất phần lớn giá trị.

---

## 10. Giai đoạn sau

- Gạch ốp tường (cần tách từng vật thể)
- Thiết bị vệ sinh (cần model 3D từng sản phẩm)
- Segmentation bằng model AI — làm đường tự động đáng tin. Giờ đã biết nó mua gì:
  **cắt vật thể, không phải render gạch**
- Lọc/gợi ý mẫu theo phòng + phong cách + ngân sách
- Zalo OA để follow-up tự động
- Cấp công cụ cho **đại lý bán lại** dùng bán cho khách của họ, mang thương hiệu mình —
  đây là vũ khí kênh phân phối, khác bài toán, cần bàn riêng

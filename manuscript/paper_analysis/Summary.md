# Paper 1

## Phân tích bài báo về Trí tuệ nhân tạo trong Sinh thiết kim nhỏ tuyến giáp

Dựa trên nguồn cung cấp, đây là phân tích bài báo:

* **Tạp chí đăng bài:** Acta Cytologica
* **Thời điểm bài đăng:** Bài báo được nhận vào ngày 18 tháng 6 năm 2020, được chấp nhận vào ngày 6 tháng 10 năm 2020 và được xuất bản trực tuyến vào ngày 16 tháng 12 năm 2020.
* **Vấn đề họ đặt ra:** Bài báo thảo luận về ứng dụng của các giải pháp Trí tuệ nhân tạo (AI) vào sinh thiết kim nhỏ tuyến giáp (FNAB). FNAB tuyến giáp là một thủ tục phổ biến, nhưng việc phân tích kết quả có thể gặp khó khăn và có thể dẫn đến kết quả không chắc chắn. AI có tiềm năng cải thiện độ chính xác và hiệu quả của việc chẩn đoán FNAB tuyến giáp.
* **Data họ có và họ dùng:** Bài báo không trình bày nghiên cứu gốc mà là **tổng hợp và phân tích các tài liệu** hiện có về ứng dụng AI trong tế bào học tuyến giáp. Các nghiên cứu được xem xét sử dụng nhiều loại dữ liệu, bao gồm:
    * **Đặc điểm hình thái của tế bào:** kích thước, hình dạng, kết cấu của nhân tế bào.
    * **Hình ảnh tế bào học:** từ các bản phết nhuộm Giemsa, Papanicolaou, và hematoxylin và eosin.
    * **Toàn bộ hình ảnh slide (WSI).**
    * **Mô tả bằng lời của các đặc điểm vi mô.**
    * **Đặc điểm lâm sàng và siêu âm.** 
* **Input, Output họ định nghĩa:** Tùy thuộc vào từng nghiên cứu, input và output được định nghĩa khác nhau. Tuy nhiên, nhìn chung, có thể tóm tắt như sau:
    * **Input:** Thông tin từ data được liệt kê ở trên.
    * **Output:** Chẩn đoán về bản chất của tổn thương tuyến giáp (lành tính hay ác tính), phân loại tổn thương tuyến giáp (ví dụ như ung thư biểu mô nhú, u nang nang, viêm tuyến giáp lymphocytic).
* **Phương pháp xử lý input ra output:** Bài báo đánh giá nhiều phương pháp AI khác nhau được sử dụng trong các nghiên cứu, bao gồm:
    * **Phân tích hình thái:** đo lường và phân tích các đặc điểm hình thái của tế bào.
    * **Mạng nơ-ron:** các loại mạng nơ-ron khác nhau, bao gồm mạng nơ-ron học sâu (deep learning).
    * **Máy vectơ hỗ trợ (SVM).**
    * **K-nearest neighbor.**
    * **Cây quyết định.**
* **Phân chia train:validation:test**: Không đề cập
* **Kết quả họ đạt được:** Bài báo cho thấy các nghiên cứu sử dụng AI trong FNAB tuyến giáp đã đạt được những kết quả đầy hứa hẹn. Một số thuật toán AI có thể phân biệt chính xác tổn thương lành tính và ác tính, thậm chí có thể phân loại các tổn thương tuyến giáp khác nhau. Tuy nhiên, **chưa có ứng dụng nào đủ mạnh để sử dụng trong lâm sàng**.
* **Kết luận họ thu được:**  Các tác giả kết luận rằng AI có tiềm năng cách mạng hóa việc chẩn đoán FNAB tuyến giáp. Tuy nhiên, vẫn còn nhiều thách thức công nghệ cần được giải quyết trước khi các ứng dụng AI sẵn sàng cho sử dụng lâm sàng.
* **Thảo luận thêm vấn đề họ đặt ra:** Bài báo nêu ra một số thách thức và hướng nghiên cứu trong tương lai, bao gồm:
    * Cần phát triển các thuật toán phù hợp với đặc thù của mẫu bệnh phẩm tuyến giáp, vì việc giải thích FNAB tuyến giáp phức tạp hơn khả năng xử lý của các công nghệ hiện tại.
    * Cần giải quyết vấn đề thời gian quét mẫu bệnh phẩm tế bào học, vì việc quét nhiều lớp ("z-stacking") để tái tạo hình ảnh ba chiều mất rất nhiều thời gian.
    * Cần giảm chi phí của máy quét kỹ thuật số và chứng minh được lợi ích kinh tế của việc sử dụng AI trong chẩn đoán FNAB tuyến giáp.

Bài báo kết thúc bằng cách khẳng định rằng khi ngày càng có nhiều nhà bệnh học tế bào học tiếp cận công nghệ mới này, chúng ta sẽ bắt đầu xác định cách các nhà bệnh học trong tương lai có thể tận dụng công nghệ AI để trả lời các câu hỏi lâm sàng thiết yếu. 

# Paper 2

## Phân tích Bài báo về Dự đoán Ung thư biểu mô tuyến giáp bằng Học sâu

Dựa trên các nguồn được cung cấp, đây là phân tích bài báo:

* **Tạp chí đăng bài:** Expert Systems With Applications
* **Thời điểm bài đăng:** Bài báo được xuất bản trực tuyến vào ngày 2 tháng 10 năm 2021 và được xuất bản trên tạp chí Expert Systems With Applications số 188 vào năm 2022.
* **Vấn đề họ đặt ra:** Bài báo tập trung vào việc phát triển một hệ thống chẩn đoán hỗ trợ máy tính (CAD) tự động để dự đoán ung thư biểu mô tuyến giáp (PTC) bằng cách **sử dụng hình ảnh tế bào học từ sinh thiết kim nhỏ (FNAC)** được xử lý bằng ThinPrep. Mục tiêu là tăng cường độ chính xác và hiệu quả trong chẩn đoán PTC, một loại ung thư tuyến giáp phổ biến nhất. (**Compare PTC & lành tính**)
* **Data họ có và họ dùng:** 
    * **Dữ liệu:** Nghiên cứu sử dụng tập dữ liệu gồm 367 hình ảnh nhuộm hematoxylin-eosin (H&E) từ các bệnh nhân trải qua FNAC và phẫu thuật cắt bỏ tuyến giáp. Trong đó, 222 trường hợp là PTC và 145 trường hợp là tổn thương lành tính (Non-PTC). Các hình ảnh này được số hóa ở độ phóng đại 400x.
    * **Cách sử dụng:** Các **hình ảnh FNAC được xử lý trước và phân đoạn thành các mảnh (fragments), mỗi mảnh chứa các cụm mô hoặc vùng quan tâm (ROI)**. Các mảnh này được sử dụng để huấn luyện, xác thực và thử nghiệm các mô hình học sâu.
* **Input, Output họ định nghĩa:**
    * **Input:** Hình ảnh F**NAC được xử lý trước và phân đoạn** thành các mảnh (**fragments**).
    * **Output:** Dự đoán xem mảnh hình ảnh là **PTC (ác tính) hay Non-PTC (lành tính)**. Dựa trên **kết quả dự đoán của từng mảnh**, nhãn dự đoán cấp độ FNAC được tính toán để đưa ra **chẩn đoán cho toàn bộ slide**.
* **Phương pháp (xử lý dữ liệu, xây dựng model) để xử lý input ra output:**
    * **Xử lý dữ liệu:** Hình ảnh FNAC được chuẩn hóa màu nhuộm để **giảm thiểu sự biến đổi màu sắc**, sau đó được **phân đoạn tự động thành các mảnh** (fragments) chứa ROI **bằng kỹ thuật phát hiện cạnh Canny** và **phương pháp đường viền**.
    * **Xây dựng mô hình:**  Sử dụng kiến trúc học sâu CNN, bao gồm **ResNet, DenseNet và Inception**. Các mô hình được huấn luyện bằng kỹ thuật transfer learning từ cơ sở dữ liệu ImageNet và data augmentation. Sau khi huấn luyện các mô hình riêng lẻ, **kỹ thuật ensemble learning** được sử dụng để **kết hợp kết quả dự đoán từ nhiều mô hình CNN và đưa ra dự đoán cuối cùng**.
* **Phân chia train:validation:test**: Họ thực hiện cắt 367 (222 PTC : 145 NonPTC) ra thành các ảnh fragments (xác định các ROIs) rồi phân chia các ảnh fragments đó ra thành 3 tập train:val:test $\to$ để dự đoán cho toàn bộ slide họ dùng công thức là: Eg: [sum(0, 0, 1, 1, 1)] / num_fragment - 0.6 is PTC. Kết quả dự đoán của 1 fragment phải đi qua tới hơn 3 model con rồi tổng hợp lại thông qua AdaBoost, vậy nên chi phí tính toán là phức tạp hơn. 
    * Huấn luyện: 980 đoạn (495 PTC và 495 không phải PTC)
    * Xác thực: 280 đoạn (140 PTC và 140 không phải PTC)
    * Kiểm tra: 140 đoạn (70 PTC và 70 không phải PTC)

* **Kết quả họ đạt được:**
    * Mô hình DenseNet161 đạt hiệu suất phân loại tốt nhất trên tập dữ liệu kiểm tra, với độ chính xác trung bình là 0.9556, độ nhạy 0.9734 và độ đặc hiệu 0.9405. 
    * Phương pháp chuẩn hóa màu nhuộm Reinhard cải thiện hiệu suất dự đoán so với các phương pháp khác.
    * Ensemble learning giúp tăng đáng kể độ chính xác của dự đoán, đạt độ chính xác lên đến 0.9971 khi sử dụng bộ phân loại AdaBoost.
    * Tuy nhiên **các kết quả này là cho ở phạm vi các fragment thôi**, chưa thực hiện đánh giá trên ảnh ở toàn mức slide. Chưa kể vì họ fragment ra các ảnh từ ảnh mức slide ban đầu rồi trộn lẫn trong các tập train, val, test $\to$ Tiềm ẩn nguy cơ kém tính độc lập giữa các tập train, val, test - dù cho họ đã tuyên bố các fragment không giao nhau, nhưng không thể tránh 1 sự thật là 1 ảnh mức slide đã bị chia vào cả 3 tập train, val, test.
* **Kết luận họ thu được:** 
    * Các mô hình học sâu có khả năng trích xuất các đặc trưng thông tin từ hình ảnh FNAC một cách hiệu quả để phân loại PTC.
    * Framework ensemble learning được đề xuất là mạnh mẽ và đầy hứa hẹn trong việc đạt được khả năng dự đoán cao.
    * Hệ thống CAD được đề xuất có tiềm năng hỗ trợ các nhà tế bào học trong chẩn đoán ung thư PTC.
* **Thảo luận thêm vấn đề họ đặt ra:**
    * Hệ thống hiện tại **chỉ giới hạn ở bài toán phân loại nhị phân (PTC và Non-PTC)** và chưa được đánh giá trên tất cả sáu loại ung thư tuyến giáp theo Hệ thống Bethesda (TBS).
    * Cần nghiên cứu thêm về việc định vị các vùng mô quan tâm vi mô khác biệt trong sáu loại TBS để cung cấp thông tin lâm sàng chi tiết hơn cho các nhà tế bào học.
    * Cần **mở rộng** kỹ thuật ensemble learning cho **bài toán phân loại sáu loại TBS**.
    * Cần nghiên cứu **ứng dụng hệ thống** trên **toàn bộ hình ảnh slide FNAC** độ phân giải cao.
    * Cần thực hiện thêm các nghiên cứu **xác thực bên ngoài** để **đánh giá hiệu suất của hệ thống trong thực tế lâm sàng**.

Bài báo kết luận bằng cách đề xuất các hướng nghiên cứu trong tương lai, bao gồm việc phát triển hệ thống chẩn đoán tự động cho sáu loại ung thư tuyến giáp TBS từ toàn bộ hình ảnh slide FNAC độ phân giải cao.

# Paper 3

## Phân tích Bài báo về Mạng Nơ-ron Xác định Ung thư biểu mô Papillary trên Phiến đồ Sinh thiết Kim nhỏ Tuyến giáp

Dựa trên các nguồn cung cấp, đây là phân tích bài báo:

* **Tạp chí đăng bài:** J Pathol Inform
* **Thời điểm bài đăng:** Bài báo được xuất bản trên J Pathol Inform 2018, 1:43.
* **Vấn đề họ đặt ra:** Bài báo nghiên cứu ứng dụng mạng nơ-ron để xác định ung thư biểu mô papillary (PTCA) trên các phiến đồ tế bào học từ sinh thiết kim nhỏ (FNAC) tuyến giáp. Việc nhận diện PTCA trên phiến đồ đôi khi gặp khó khăn, đặc biệt khi các phiến đồ chỉ hiển thị các tế bào lành tính. Bài báo nhằm mục đích sử dụng mạng nơ-ron, một mô hình học máy đã được chứng minh là hiệu quả trong nhận dạng hình ảnh, để phân loại các phiến đồ tế bào học tuyến giáp.
* **Data họ có và họ dùng:**
    * **Dữ liệu:**  Nghiên cứu sử dụng 370 ảnh chụp hiển vi từ các phiến đồ nhuộm Romanowsky/Pap của PTCA và các tổn thương không phải PTCA. Trong đó, 186 ảnh từ các phiến đồ PTCA và 184 ảnh từ các phiến đồ không phải PTCA. Dữ liệu được chia thành tập huấn luyện và tập kiểm tra.
    * **Cách sử dụng:**  Các ảnh chụp hiển vi được sử dụng để huấn luyện mạng nơ-ron. Mạng nơ-ron được huấn luyện với nhiều loại hình ảnh, bao gồm cả ảnh chụp ở độ phóng đại 10x và 40x. Các trường hợp ranh giới không được đưa vào nghiên cứu này.
* **Input, Output họ định nghĩa:**
    * **Input:** Các ảnh chụp hiển vi từ các phiến đồ tế bào học tuyến giáp. Mỗi ảnh có ba kênh màu đỏ, xanh lá cây và xanh dương, đại diện cho độ sâu của hình ảnh.
    * **Output:** Phân loại ảnh chụp hiển vi là PTCA hoặc không phải PTCA.
* **Phương pháp (xử lý dữ liệu, xây dựng model) để xử lý input ra output:**
    * **Xử lý dữ liệu:** Các ảnh chụp hiển vi được cắt xén để tập trung vào các vùng quan tâm.  Nhiều ảnh chụp hiển vi từ một phiến đồ được chụp để đảm bảo mạng nơ-ron được huấn luyện với nhiều loại hình ảnh.
    * **Xây dựng mô hình:** Mạng nơ-ron tích chập được sử dụng. Mạng nơ-ron này được xây dựng dựa trên kiến trúc mạng nơ-ron sâu, lấy cảm hứng từ đường dẫn thị giác bụng của não linh trưởng.  Mạng nơ-ron bao gồm nhiều lớp được kết nối theo cách thức feedforward, mỗi lớp bao gồm nhiều nơ-ron. Các lớp đại diện cho một vùng võng mạc cụ thể của não.  Mạng nơ-ron được huấn luyện bằng cách sử dụng thư viện TensorFlow và Keras. Quá trình huấn luyện bao gồm nhiều epoch, trong đó mạng nơ-ron học cách phân loại các hình ảnh dựa trên các đặc trưng được trích xuất từ các lớp trước đó. Sau mỗi epoch, độ chính xác trên tập kiểm tra được đo đồng thời.
* **Phân chia train:validation:test**: 
  * Tập train: 370 = (x10: 92 + 89, x40: 68 + 74)
  * Tập test: 48 ảnh khác = (x10: 14 + 14, x40: 10 + 10)
  * Tập val: Không có
* **Kết quả họ đạt được:**
    * Tóm lược lại là được 82.76% chính xác cho việc dự đoán.
    * Ngoài ra họ liệt kê nhiều cái khác nhưng kết quả chênh lệch lớn. 
* **Kết luận họ thu được:**
    * Mạng nơ-ron có tiềm năng xác định PTCA trên các phiến đồ tế bào học.
    * Cần huấn luyện thêm với tập dữ liệu lớn hơn và đa dạng hơn để cải thiện hiệu suất của mạng nơ-ron.
* **Thảo luận thêm vấn đề họ đặt ra:**
    * Các phiến đồ tế bào học có thể có nhiều biến thể, điều này gây khó khăn cho mạng nơ-ron trong việc học các đặc trưng phân biệt.
    * Việc huấn luyện mạng nơ-ron với tập dữ liệu lớn và đa dạng là rất quan trọng để cải thiện hiệu suất.
    * Cần phát triển các kỹ thuật chụp ảnh tự động và xử lý phiến đồ để tạo ra một kho lưu trữ hình ảnh bệnh lý lớn và đa dạng.

Bài báo kết luận bằng cách nhấn mạnh sự cần thiết của việc huấn luyện thêm với các tập dữ liệu lớn hơn và đa dạng hơn để cải thiện hiệu suất của mạng nơ-ron trong việc xác định PTCA trên các phiến đồ tế bào học.

# Paper 4

## Phân tích bài báo y khoa về chẩn đoán bướu giáp

- **Tạp chí đăng bài:** Lancet Digit Health
- **Thời điểm bài đăng:** Đăng trực tuyến ngày 6 tháng 6 năm 2024
- **Vấn đề họ đặt ra:** Chẩn đoán chính xác bướu giáp **ác tính và lành tính** thông qua tế bào học bằng kim nhỏ (FNA) là rất quan trọng để can thiệp điều trị thích hợp. Tuy nhiên, việc chẩn đoán tế bào học rất tốn thời gian và bị cản trở bởi tình trạng thiếu các nhà tế bào học giàu kinh nghiệm. Các công cụ hỗ trợ đáng tin cậy có thể cải thiện hiệu quả và độ chính xác của chẩn đoán tế bào học.
- **Dữ liệu họ có và họ dùng:**
    - **11.254 ảnh toàn bộ slide (WSI) từ 4.037 bệnh nhân** được sử dụng để huấn luyện mô hình học sâu.
    - **Bộ dữ liệu hồi cứu** gồm 5.638 WSI của 2.914 bệnh nhân từ bốn trung tâm y tế được sử dụng để xác thực.
    - 469 bệnh nhân được tuyển dụng cho nghiên cứu triển vọng về hiệu suất của mô hình AI và 537 mẫu bướu giáp của họ đã được sử dụng.
    - Các nhóm huấn luyện và xác thực được ghi danh từ ngày 1 tháng 1 năm 2016 đến ngày 1 tháng 8 năm 2022 và bộ dữ liệu tiềm năng được tuyển dụng từ ngày 1 tháng 8 năm 2022 đến ngày 1 tháng 1 năm 2023.
- **Input:** **Ảnh toàn bộ slide (WSI)** của các mẫu tế bào học FNA bướu giáp.
- **Output:** Chẩn đoán bướu giáp dựa trên Hệ thống Báo cáo Bethesda về Tế bào học tuyến giáp (TBSRTC).
- **Phương pháp xử lý input ra output:**
    - Xây dựng **hệ thống ThyroPower** sử dụng **học sâu**, bao gồm:
        - Mạng PAGIN để **trích xuất các đặc trưng ở cấp độ tế bào** và xác định vị trí bất thường.
        - Mô hình **phân loại cấp WSI (Random Forest và TNF)** để đưa ra quyết định chẩn đoán cuối cùng.
- **Phân chia tập dữ liệu:** 
    - Tập huấn luyện: 11.254 WSI từ 4.037 bệnh nhân tại Bệnh viện Sun Yat-sen Memorial, Đại học Sun Yat-sen.
    - Tập xác thực:
        - Xác thực nội bộ: từ Bệnh viện Sun Yat-sen Memorial, Đại học Sun Yat-sen.
        - Xác thực bên ngoài: từ ba trung tâm y tế:
            - Viện & Bệnh viện Ung thư Tứ Xuyên
            - Bệnh viện Nhân dân số 1 Phật Sơn
            - Bệnh viện trực thuộc thứ ba, Đại học Y khoa Quảng Châu
    - **Không có tập kiểm tra riêng biệt được đề cập.**
- **Kết quả đạt được:**
    - **Không có kết quả accuracy hoặc F1-score trên tập kiểm tra được đề cập.** 
    - Có cung cấp **F2-score là 0.847**.
    - **AUROC cho TBSRTC III** + (phân biệt lành tính với TBSRTC III, IV, V và VI) là **0,930** (KTC 95% 0,921–0,939) cho xác thực nội bộ SYSMH và 0,944 (0,929–0,959), 0,939 (0,924–0,955), 0,971 (0,938–1,000) cho FPHF, SCHI và TAHGMU, tương ứng.
    - **AUROC cho TBSRTC V** + (phân biệt lành tính với TBSRTC V và VI) là **0,990** (KTC 95% 0,986–0,995) cho xác thực nội bộ SYSMH và 0,988 (0,980–0,995), 0,965 (0,953–0,977) và 0,991 (0,972–1,000) cho FPHF, SCHI và TAHGMU, tương ứng.
    - Đối với nghiên cứu triển vọng tại SYSMH, AUROC của TBSRTC III+ và TBSRTC V+ lần lượt là 0,977 và 0,981.
- **Kết luận:** Nghiên cứu đã phát triển mô hình hỗ trợ AI có tên là hệ thống Nhận dạng tập hợp WSI định hướng bản vá tuyến giáp (ThyroPower), tạo điều kiện chẩn đoán tế bào học nhanh chóng và mạnh mẽ các nốt tuyến giáp, có khả năng nâng cao khả năng chẩn đoán của các nhà tế bào học. Hơn nữa, nó đóng vai trò như một giải pháp tiềm năng để giảm bớt tình trạng khan hiếm các nhà tế bào học.
- **Thảo luận thêm vấn đề:**
    - Cần thu thập thêm dữ liệu huấn luyện để cải thiện độ chính xác chẩn đoán cho các trường hợp SFN (nghi ngờ u nang tuyến giáp).
    - Tỷ lệ hiện mắc của bộ dữ liệu có ảnh hưởng đáng kể đến giá trị dự đoán (NPV và PPV).
    - Cần thu thập thêm mẫu có thông tin xét nghiệm phân tử để huấn luyện hệ thống AI nhằm đạt được dự đoán nhạy và đặc hiệu hơn đối với các mẫu AUS (không điển hình chưa xác định được ý nghĩa).
    - Hệ thống ThyroPower được thiết kế để hỗ trợ chứ không thay thế các chuyên gia con người.

# Paper 5

## Phân tích Bài báo về Ứng dụng AI trong Chẩn đoán Tuyến giáp

* **Tạp chí đăng bài**: Cancers
* **Thời điểm bài đăng**: 24 tháng 1 năm 2023
* **Vấn đề họ đặt ra**: Bài báo tập trung vào việc đánh giá các ứng dụng mới nhất của trí tuệ nhân tạo (AI) trong chẩn đoán và phân loại bướu cổ giáp, đặc biệt là trong lĩnh vực siêu âm và chẩn đoán vi thể.

## Các Nghiên cứu về Cytopathology trong Chẩn đoán Bướu Cổ giáp

Bài báo đã đề cập đến một số nghiên cứu sử dụng AI trong việc phân tích ảnh tế bào học (cytopathology). Dưới đây là tóm tắt chi tiết về từng nghiên cứu:

**1. Nghiên cứu của Guan và cộng sự (2019)**

* **Mục tiêu:** Phân biệt ung thư biểu mô tuyến giáp thể nhú (**PTC**) với bướu **lành tính** dựa trên ảnh tế bào học từ FNA.
* **Dữ liệu:**
    * 887 ảnh từ 279 bệnh nhân. 
    * 759 ảnh huấn luyện (407 PTC, 352 lành tính).
    * 128 ảnh kiểm tra (69 PTC, 59 lành tính).
* **Input:** Ảnh tế bào học.
* **Output:** Phân loại PTC hoặc lành tính.
* **Phương pháp:** Sử dụng hai thuật toán CNN là VGC-16 và Inception-v3.
* **Kết quả:** 
    * VGC-16: Độ nhạy 100%, độ đặc hiệu 94.91%.
    * Inception-v3: Độ nhạy 98.55%, độ đặc hiệu 86.44%.
* **Hạn chế:** Nghiên cứu chưa so sánh hiệu suất của AI với chuyên gia.

**2. Nghiên cứu của Sanyal và cộng sự (2018)**

* **Mục tiêu:** Phân biệt PTC với ung thư tuyến giáp không phải PTC.
* **Dữ liệu:**
    * 544 ảnh từ 20 bệnh nhân.
    * 370 ảnh huấn luyện (184 PTC, 186 không phải PTC).
    * 174 ảnh kiểm tra (42 PTC, 132 không phải PTC) ở hai mức phóng đại x10 và x40.
* **Input:** Ảnh vi mô của vùng quan tâm trên phiến đồ tế bào học FNA.
* **Output:** Phân loại PTC hoặc không phải PTC.
* **Phương pháp:** Sử dụng thuật toán CNN.
* **Kết quả:** 
    * Độ nhạy 90.48%, độ đặc hiệu 83.33%, độ chính xác 85.06% khi yêu cầu phân loại đúng ở ít nhất một mức phóng đại.
    * Độ nhạy 33.33%, độ đặc hiệu 98.48%, độ chính xác 82.76% khi yêu cầu phân loại đúng ở cả hai mức phóng đại.
* **Hạn chế:** Nghiên cứu chưa so sánh hiệu suất của AI với chuyên gia.

**3. Nghiên cứu của Elliott và cộng sự (2020)**

* **Mục tiêu:** Phát hiện ROI (vùng quan tâm) và phân loại bướu lành tính/ác tính trên ảnh WSI từ FNA.
* **Dữ liệu:** 908 ảnh WSI từ 659 bệnh nhân.
* **Input:** Ảnh WSI.
* **Output:**
    * Phân loại ROI/không phải ROI.
    * Phân loại bướu lành tính/ác tính.
    * Phân loại theo hệ thống Bethesda (TBSRTC).
* **Phương pháp:** Sử dụng thuật toán CNN.
* **Kết quả:** 
    * AUC cho việc phân biệt ROI/không phải ROI: 0.985.
    * Độ nhạy 92%, độ đặc hiệu 90.5% trong việc phân loại bướu lành tính/ác tính.
    * AUC cho độ chính xác của AI: 0.932.
    * AUC cho độ chính xác của chuyên gia: 0.931.
* **Điểm nổi bật:** Nghiên cứu đã kết hợp AI và đánh giá của chuyên gia để cải thiện độ đặc hiệu và AUC.

**4. Nghiên cứu của Dov và cộng sự (2022)**

* **Mục tiêu:** Đánh giá khả năng của AI trong việc phát hiện ROI trên ảnh WSI từ FNA để hỗ trợ chẩn đoán.
* **Dữ liệu:** 908 ảnh WSI từ 659 bệnh nhân.
* **Input:** Ảnh WSI.
* **Output:** 100 ảnh ROI chứa nhóm tế bào nang.
* **Phương pháp:** Sử dụng thuật toán CNN để phát hiện ROI. Chuyên gia sau đó đánh giá 100 ảnh ROI này.
* **Kết quả:**
    * Độ tương đồng (k) giữa đánh giá dựa trên WSI đầy đủ và 100 ảnh ROI do AI lựa chọn: 0.924 cho TBSRTC, 0.834 cho phân loại nguy cơ.
    * Độ tương đồng (k) giữa đánh giá dựa trên WSI đầy đủ và kết quả giải phẫu bệnh: 0.845 cho TBSRTC, 0.669 cho phân loại nguy cơ.
* **Kết luận:** AI có thể hỗ trợ hiệu quả trong việc lựa chọn ROI, giúp chuyên gia tiết kiệm thời gian và tập trung vào các vùng quan trọng trên phiến đồ.

# Paper 6


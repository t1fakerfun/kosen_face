import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Classifier',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  final picker = ImagePicker();
  final TextEditingController _labelController = TextEditingController();
  String _result = "ここに結果が表示されます";
  bool _isLoading = false;

  // --- サーバー設定 ---
  // Androidエミュレータは '10.0.2.2'
  // iOSシミュレータは '127.0.0.1'
  // 実機の場合はPCのIPアドレス (例: '192.168.1.10' など)
  final String serverUrl = Platform.isAndroid ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';

  // 画像を選択する関数
  Future<void> _pickImage() async {
      print("--- 1. ボタンが押されました ---"); // これが出るか？

      try {
        final pickedFile = await picker.pickImage(source: ImageSource.gallery);

        print("--- 2. pickImageが完了しました ---"); // これが出るか？

        if (pickedFile != null) {
          print("--- 3. 画像が選択されました: ${pickedFile.path} ---");
          setState(() {
            _image = File(pickedFile.path);
            _result = "画像を選択しました";
          });
        } else {
          print("--- 3. キャンセルされました ---");
        }
      } catch (e) {
        print("--- エラー発生!!! ---");
        print(e); // ここに詳細なエラーが出るはず
      }
    }

  // サーバーへ画像を送信して予測する関数 (/predict)
  Future<void> _predictImage() async {
    if (_image == null) {
      setState(() => _result = "まずは画像を選択してください");
      return;
    }

    setState(() => _isLoading = true);

    try {
      var uri = Uri.parse('$serverUrl/predict');
      var request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', _image!.path));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        // 日本語で読みやすく整形
        String label = data['predicted_label'];
        Map<String, dynamic> probs = data['probabilities'] ?? {};

        String probText = probs.entries.map((e) {
          double val = e.value * 100;
          return "${e.key}: ${val.toStringAsFixed(1)}%";
        }).join("\n");

        setState(() {
          _result = "予測結果: $label\n\n$probText";
        });
      } else {
        setState(() => _result = "サーバーエラー: ${response.statusCode}");
      }
    } catch (e) {
      setState(() => _result = "通信エラー: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // サーバーへ画像とラベルを送信して学習させる関数 (/register)
  Future<void> _registerImage() async {
    if (_image == null) {
      setState(() => _result = "まずは画像を選択してください");
      return;
    }
    if (_labelController.text.isEmpty) {
      setState(() => _result = "正解ラベル(グループ名)を入力してください");
      return;
    }

    setState(() => _isLoading = true);

    try {
      var uri = Uri.parse('$serverUrl/register');
      var request = http.MultipartRequest('POST', uri);

      request.files.add(await http.MultipartFile.fromPath('file', _image!.path));
      request.fields['label'] = _labelController.text;

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        setState(() {
          _result = "登録完了: ${data['message']}";
        });
      } else {
        setState(() => _result = "登録失敗: ${response.body}");
      }
    } catch (e) {
      setState(() => _result = "通信エラー: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Python Server Test')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // 画像表示エリア
            Container(
              height: 200,
              width: double.infinity,
              color: Colors.grey[200],
              child: _image == null
                  ? const Center(child: Text('画像が選択されていません'))
                  : Image.file(_image!, fit: BoxFit.contain),
            ),
            const SizedBox(height: 10),

            // 画像選択ボタン
            ElevatedButton.icon(
              onPressed: _pickImage,
              icon: const Icon(Icons.photo_library),
              label: const Text("ギャラリーから画像を選択"),
            ),

            const Divider(height: 30, thickness: 2),

            // 予測ボタンエリア
            const Text("【機能1: 判定】", style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 5),
            ElevatedButton(
              onPressed: _isLoading ? null : _predictImage,
              style: ElevatedButton.styleFrom(backgroundColor: Colors.blue, foregroundColor: Colors.white),
              child: const Text("画像を判定する (Predict)"),
            ),

            const Divider(height: 30, thickness: 2),

            // 学習ボタンエリア
            const Text("【機能2: 追加学習】", style: TextStyle(fontWeight: FontWeight.bold)),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8.0),
              child: TextField(
                controller: _labelController,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: '正解ラベル (例: mechanical)',
                  hintText: 'フォルダ名になります',
                ),
              ),
            ),
            ElevatedButton(
              onPressed: _isLoading ? null : _registerImage,
              style: ElevatedButton.styleFrom(backgroundColor: Colors.green, foregroundColor: Colors.white),
              child: const Text("学習データとして登録 (Register)"),
            ),

            const Divider(height: 30, thickness: 2),

            // 結果表示エリア
            if (_isLoading) const CircularProgressIndicator(),
            const SizedBox(height: 10),
            Text(
              _result,
              style: const TextStyle(fontSize: 16),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

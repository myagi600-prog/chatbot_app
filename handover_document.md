# AIチャットボット 引継ぎ資料

## 1. 基本的な使用サービス情報

*   **AIモデル**: Google Gemini (使用モデル: `models/gemini-pro-latest`)
*   **Webフレームワーク**: Streamlit
*   **デプロイ環境**: Streamlit Community Cloud
*   **RAG関連ライブラリ**:
    *   LangChain (RAGフレームワーク)
    *   HuggingFaceEmbeddings (Embeddingモデル: `paraphrase-multilingual-MiniLM-L12-v2` を利用)
    *   DocArrayInMemorySearch (インメモリベクトルストア)
*   **ファイル処理ライブラリ**:
    *   PyPDFLoader (PDFファイル読み込み)
    *   Docx2txtLoader (Wordファイル読み込み)
    *   TextLoader (テキストファイル読み込み)
    *   UnstructuredExcelLoader (Excelファイル読み込み、`unstructured[xlsx]`パッケージを使用)

## 2. 開発内容

本プロジェクトでは、独自のファイル（PDF, Word, Excel, TXT）を知識ベースとして利用できるAIチャットボットの開発を行いました。主な開発内容は以下の通りです。

*   **AIチャットボットの基本機能実装**: Google Gemini APIと連携したチャット機能およびRAG（Retrieval-Augmented Generation）機能を実装しました。
*   **Excelファイル処理の安定化**: 従来の`openpyxl`による手動読み込み処理で発生していたフロントエンドのクラッシュ問題に対し、より堅牢な`unstructured`ライブラリ（`UnstructuredExcelLoader`）を導入することで解決しました。
*   **Geminiモデル名のデプロイ環境への適合**: デプロイ環境で`gemini-1.5-flash`モデルが利用できない問題が発生したため、利用可能なモデルを特定し、`gemini-pro`を経て最終的に`models/gemini-pro-latest`モデルを使用するように修正しました。
*   **知識ベースの追加機能**: 既存の知識ベースに新しいファイルの内容を追加（マージ）できる機能を実装しました。これにより、ユーザーは複数のファイルを段階的に知識ベースに組み込むことが可能になりました。
*   **知識ベースのクリア機能**: 知識ベースの内容をリセットするための「知識ベースをクリア」ボタンをサイドバーに追加しました。
*   **ファイルアップローダーの表示テキストと対応ファイルタイプの調整**: ユーザーの要望に基づき、ファイルアップローダーの表示テキストを「知識ファイル（メモ帳/Word/Excelなど）をアップロード」に変更し、対応ファイルタイプを`.txt`, `.xlsx`, `.xls`, `.docx`に絞り込みました。
*   **チャットタイトルの変更**: アプリケーションのタイトルを「SmartAssistant」に変更しました。
*   **画像ファイルのRAG対応**: 一時的に画像処理ロジックの追加を試みましたが、実用性の観点から見送られました。

## 3. 現在の状況

*   WebアプリはStreamlit Community Cloudにデプロイ済みです。
*   Gemini APIを利用したチャット機能は正常に動作しています。
*   テキストファイル、Wordファイル、Excelファイルを知識ベースとしてアップロードし、その内容に基づいてAIが回答するRAG機能は正常に動作しています。
*   複数のファイルをアップロードして既存の知識ベースに追加する機能も正常に動作します。
*   知識ベースをクリアする機能も正常に動作します。
*   チャットタイトルは「SmartAssistant」です。
*   ファイルアップローダーは、メモ帳（.txt）、Word（.docx）、Excel（.xlsx, .xls）ファイルに対応しています。
*   知識ベースクリア後に会話を開始しようとするとエラーが発生する問題は、`vector_store`が`None`の場合に適切な警告を表示するように修正され、解消済みです。

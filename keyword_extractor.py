"""
关键词提取器：从输入文本中提取中心名词和关键词

支持多种提取方法：
传统方法：
- spacy: 基于词性标注和命名实体识别
- nltk: 基于词性标注和停用词过滤  
- yake: 基于YAKE关键词提取算法

深度学习方法：
- bert: 基于BERT模型的掩码语言建模
- transformer: 基于Transformer的语义理解
- keybert: 基于BERT embeddings的关键词提取
- llm: 基于大语言模型(GPT/Claude)的智能提取

示例：
    "A woman holding an umbrella" -> 中心词: "woman"
    "A yellow hat carried by a man" -> 中心词: "hat"
"""

import re
import os
import warnings
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略一些警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class KeywordExtractor:
    """关键词提取器类"""
    
    def __init__(self, method: str = 'nltk'):
        """
        初始化关键词提取器
        
        Args:
            method (str): 提取方法，可选:
                传统方法: 'spacy', 'nltk', 'yake'
                深度学习方法: 'bert', 'transformer', 'keybert', 'llm'
        """
        self.method = method.lower()
        
        # 传统方法相关
        self.nlp = None
        self.yake_extractor = None
        
        # 深度学习方法相关
        self.bert_tokenizer = None
        self.bert_model = None
        self.sentence_transformer = None
        self.keybert_model = None
        self.llm_client = None
        
        # 定义停用词集合
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
            'don', 'should', 'now', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'
        }
        
        # 初始化相应的NLP工具
        self._initialize_nlp_tools()
        
    def _initialize_nlp_tools(self):
        """初始化NLP工具"""
        # spaCy在多个方法中需要使用
        if self.method in ['spacy', 'transformer']:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy模型未找到，尝试下载...")
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                    self.nlp = spacy.load("en_core_web_sm")
            except ImportError:
                logger.error("spaCy未安装，请运行: pip install spacy")
                raise
        
        # 根据具体方法初始化相应工具        
        if self.method == 'nltk':
            try:
                import nltk
                # 下载必要的数据
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                from nltk.corpus import stopwords
                english_stops = set(stopwords.words('english'))
                self.stop_words.update(english_stops)
            except ImportError:
                logger.error("NLTK未安装，请运行: pip install nltk")
                raise
                
        if self.method == 'yake':
            try:
                import yake
                self.yake_extractor = yake.KeywordExtractor(
                    lan="en",
                    n=3,  # n-gram大小
                    dedupLim=0.7,  # 去重阈值
                    top=10  # 返回前10个关键词
                )
            except ImportError:
                logger.error("YAKE未安装，请运行: pip install yake")
                raise
                
        # 深度学习方法初始化
        if self.method == 'bert':
            try:
                from transformers import pipeline
                import torch
                
                logger.info("初始化BERT pipeline...")
                # 使用pipeline API更简单和稳定
                self.bert_pipeline = pipeline('fill-mask', model='bert-base-uncased')
                logger.info("BERT pipeline已初始化")
                
            except ImportError:
                logger.error("Transformers未安装，请运行: pip install transformers torch")
                raise
                
        if self.method == 'transformer':
            try:
                from sentence_transformers import SentenceTransformer
                
                model_name = "all-MiniLM-L6-v2"  # 轻量级但效果好的模型
                logger.info(f"加载Sentence Transformer模型: {model_name}")
                
                self.sentence_transformer = SentenceTransformer(model_name)
                logger.info("Sentence Transformer模型已加载")
                
            except ImportError:
                logger.error("Sentence Transformers未安装，请运行: pip install sentence-transformers")
                raise
                
        if self.method == 'keybert':
            try:
                from keybert import KeyBERT
                
                logger.info("初始化KeyBERT模型")
                self.keybert_model = KeyBERT()
                logger.info("KeyBERT模型已初始化")
                
            except ImportError:
                logger.error("KeyBERT未安装，请运行: pip install keybert")
                raise
                
        if self.method == 'llm':
            # 大语言模型方法 - 可以集成OpenAI GPT, Anthropic Claude等
            logger.info("初始化大语言模型接口")
            self.llm_client = self._initialize_llm_client()
        
        # 验证方法是否被支持
        supported_methods = ['spacy', 'nltk', 'yake', 'bert', 'transformer', 'keybert', 'llm']
        if self.method not in supported_methods:
            raise ValueError(f"不支持的提取方法: {self.method}，支持的方法: {supported_methods}")
    
    def _initialize_llm_client(self):
        """初始化大语言模型客户端"""
        # 这里可以根据需要选择不同的LLM服务
        # 示例：OpenAI GPT
        try:
            import openai
            
            # 检查是否有API密钥
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key
                logger.info("OpenAI客户端已初始化")
                return openai
            else:
                logger.warning("未找到OPENAI_API_KEY环境变量，LLM方法将使用本地规则")
                return None
                
        except ImportError:
            logger.warning("OpenAI包未安装，请运行: pip install openai")
            return None
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        提取关键词
        
        Args:
            text (str): 输入文本
            max_keywords (int): 最大关键词数量
            
        Returns:
            List[str]: 关键词列表
        """
        if not text.strip():
            return []
            
        # 传统方法
        if self.method == 'spacy':
            return self._extract_keywords_spacy(text, max_keywords)
        elif self.method == 'nltk':
            return self._extract_keywords_nltk(text, max_keywords)
        elif self.method == 'yake':
            return self._extract_keywords_yake(text, max_keywords)
        # 深度学习方法
        elif self.method == 'bert':
            return self._extract_keywords_bert(text, max_keywords)
        elif self.method == 'transformer':
            return self._extract_keywords_transformer(text, max_keywords)
        elif self.method == 'keybert':
            return self._extract_keywords_keybert(text, max_keywords)
        elif self.method == 'llm':
            return self._extract_keywords_llm(text, max_keywords)
        else:
            raise ValueError(f"不支持的提取方法: {self.method}")
    
    def extract_main_subject(self, text: str) -> str:
        """
        提取文本的主要中心词(通常是主语名词)
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 中心词
        """
        if not text.strip():
            return ""
            
        # 传统方法
        if self.method == 'spacy':
            return self._extract_main_subject_spacy(text)
        elif self.method == 'nltk':
            return self._extract_main_subject_nltk(text)
        elif self.method == 'yake':
            return self._extract_main_subject_yake(text)
        # 深度学习方法
        elif self.method == 'bert':
            return self._extract_main_subject_bert(text)
        elif self.method == 'transformer':
            return self._extract_main_subject_transformer(text)
        elif self.method == 'keybert':
            return self._extract_main_subject_keybert(text)
        elif self.method == 'llm':
            return self._extract_main_subject_llm(text)
        else:
            raise ValueError(f"不支持的提取方法: {self.method}")
    
    def _extract_keywords_spacy(self, text: str, max_keywords: int) -> List[str]:
        """使用spaCy提取关键词"""
        doc = self.nlp(text)
        keywords = []
        
        # 提取名词和形容词，优先考虑名词
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 1 and
                token.text.lower() not in self.stop_words):
                keywords.append(token.lemma_.lower())
        
        # 去重并保持顺序
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:max_keywords]
    
    def _extract_keywords_nltk(self, text: str, max_keywords: int) -> List[str]:
        """使用NLTK提取关键词"""
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.tag import pos_tag
        from nltk.stem import WordNetLemmatizer
        
        lemmatizer = WordNetLemmatizer()
        
        # 分词和词性标注
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        keywords = []
        # 提取名词和形容词，优先考虑名词
        for word, pos in pos_tags:
            if (pos.startswith('NN') or pos.startswith('JJ')) and \
               word not in self.stop_words and \
               len(word) > 1 and \
               word.isalpha():
                lemma = lemmatizer.lemmatize(word, 'n' if pos.startswith('NN') else 'a')
                keywords.append(lemma)
        
        # 去重并保持顺序
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:max_keywords]
    
    def _extract_keywords_yake(self, text: str, max_keywords: int) -> List[str]:
        """使用YAKE提取关键词"""
        keywords_scores = self.yake_extractor.extract_keywords(text)
        
        # 提取关键词，按分数排序（分数越低越重要）
        keywords = []
        for keyword, score in keywords_scores[:max_keywords]:
            # 过滤停用词和短词
            words = keyword.lower().split()
            filtered_words = [w for w in words if w not in self.stop_words and len(w) > 1]
            if filtered_words:
                keywords.extend(filtered_words)
        
        # 去重
        return list(dict.fromkeys(keywords))[:max_keywords]
    
    def _extract_main_subject_spacy(self, text: str) -> str:
        """使用spaCy提取主要中心词"""
        doc = self.nlp(text)
        
        # 优先寻找主语
        for token in doc:
            if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PROPN"]:
                return token.lemma_.lower()
        
        # 如果没有找到主语，寻找第一个重要名词
        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN"] and 
                not token.is_stop and 
                len(token.text) > 1):
                return token.lemma_.lower()
        
        # 最后尝试从关键词中选择第一个
        keywords = self._extract_keywords_spacy(text, 5)
        return keywords[0] if keywords else ""
    
    def _extract_main_subject_nltk(self, text: str) -> str:
        """使用NLTK提取主要中心词"""
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.tag import pos_tag
        from nltk.stem import WordNetLemmatizer
        
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # 寻找第一个名词作为中心词
        for word, pos in pos_tags:
            if (pos.startswith('NN') and 
                word not in self.stop_words and 
                len(word) > 1 and 
                word.isalpha()):
                return lemmatizer.lemmatize(word, 'n')
        
        # 如果没有找到名词，从关键词中选择
        keywords = self._extract_keywords_nltk(text, 5)
        return keywords[0] if keywords else ""
    
    def _extract_main_subject_yake(self, text: str) -> str:
        """使用YAKE提取主要中心词"""
        # YAKE主要用于关键词提取，对于主词提取，我们使用简单的启发式方法
        keywords = self._extract_keywords_yake(text, 10)
        
        # 在关键词中寻找名词
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.tag import pos_tag
        
        # 对每个关键词进行词性标注，找出名词
        for keyword in keywords:
            tokens = word_tokenize(keyword.lower())
            pos_tags = pos_tag(tokens)
            for word, pos in pos_tags:
                if pos.startswith('NN') and len(word) > 1:
                    return word
        
        # 如果没有找到名词，返回第一个关键词
        return keywords[0] if keywords else ""
    
    # ==================== 深度学习方法实现 ====================
    
    def _extract_keywords_bert(self, text: str, max_keywords: int) -> List[str]:
        """使用BERT模型提取关键词"""
        try:
            import torch
            from transformers import pipeline
            
            # 如果还没有初始化BERT pipeline，则创建
            if not hasattr(self, 'bert_pipeline'):
                self.bert_pipeline = pipeline('fill-mask', model='bert-base-uncased')
            
            # 分词并过滤
            words = text.lower().split()
            filtered_words = [w.strip('.,!?;:"()[]') for w in words 
                            if w.lower() not in self.stop_words and len(w) > 1]
            
            if not filtered_words:
                return []
            
            keyword_scores = []
            
            # 对每个候选词计算重要性分数
            for word in filtered_words:
                try:
                    # 创建掩码版本的文本
                    masked_text = text.replace(word, '[MASK]', 1)
                    
                    # 使用BERT pipeline预测
                    predictions = self.bert_pipeline(masked_text)
                    
                    # 查找原词在预测结果中的分数
                    word_score = 0.0
                    for pred in predictions:
                        pred_token = pred['token_str'].lower().strip()
                        if pred_token == word.lower() or word.lower().startswith(pred_token):
                            word_score = pred['score']
                            break
                    
                    # 如果找不到精确匹配，给一个基于语义相似性的分数
                    if word_score == 0.0:
                        # 简单的启发式分数：词长度和在句子中的位置
                        word_score = min(len(word) / 10.0, 0.5)
                    
                    keyword_scores.append((word.lower(), word_score))
                    
                except Exception as word_error:
                    # 单个词处理失败时，给一个默认分数
                    logger.debug(f"处理词 '{word}' 时出错: {word_error}")
                    keyword_scores.append((word.lower(), 0.01))
            
            # 按分数排序
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 提取关键词并去重
            seen = set()
            keywords = []
            for word, score in keyword_scores:
                if word not in seen and len(keywords) < max_keywords:
                    seen.add(word)
                    keywords.append(word)
            
            return keywords
            
        except Exception as e:
            logger.warning(f"BERT关键词提取失败: {e}，回退到NLTK方法")
            return self._extract_keywords_nltk(text, max_keywords)
    
    def _extract_keywords_transformer(self, text: str, max_keywords: int) -> List[str]:
        """使用Sentence Transformer提取关键词"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 第一步：使用spaCy进行词性标注，提取名词
            doc = self.nlp(text)
            
            # 提取名词作为候选词
            noun_candidates = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN']) and \
                   not token.is_stop and \
                   not token.is_punct and \
                   len(token.text) > 1 and \
                   token.text.lower() not in self.stop_words and \
                   token.text.isalpha():
                    lemma = token.lemma_.lower()
                    noun_candidates.append(lemma)
            
            # 如果没有提取到名词，回退到简单分词方法
            if not noun_candidates:
                words = text.lower().split()
                noun_candidates = [w.strip('.,!?;:"()[]') for w in words 
                                 if w.lower() not in self.stop_words and len(w) > 1]
            
            if not noun_candidates:
                return []
            
            # 去重保持候选词的唯一性
            unique_candidates = []
            seen = set()
            for candidate in noun_candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    unique_candidates.append(candidate)
            
            # 第二步：计算嵌入向量
            # 获取文本整体的语义向量
            text_embedding = self.sentence_transformer.encode([text])
            
            # 获取每个候选名词的语义向量
            candidate_embeddings = self.sentence_transformer.encode(unique_candidates)
            
            # 第三步：计算余弦相似度
            # 计算每个候选词与整体文本的语义相似度
            similarities = cosine_similarity(candidate_embeddings, text_embedding).flatten()
            
            # 按相似度排序
            word_scores = list(zip(unique_candidates, similarities))
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 提取最终关键词
            keywords = []
            for word, score in word_scores:
                if len(keywords) < max_keywords:
                    keywords.append(word)
            
            return keywords
            
        except Exception as e:
            logger.warning(f"Transformer关键词提取失败: {e}，回退到NLTK方法")
            return self._extract_keywords_nltk(text, max_keywords)
    
    def _extract_keywords_keybert(self, text: str, max_keywords: int) -> List[str]:
        """使用KeyBERT提取关键词"""
        try:
            # 使用KeyBERT提取关键词
            keywords_scores = self.keybert_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2),  # 1-2个词的短语
                stop_words='english',
                top_n=max_keywords,  # 修正参数名
                use_maxsum=True,  # 使用MaxSum算法提高多样性
                nr_candidates=20,
                use_mmr=True,     # 使用MMR算法平衡相关性和多样性
                diversity=0.5
            )
            
            # 提取关键词（忽略分数）
            keywords = []
            for keyword, score in keywords_scores:
                # 将短语分解为单词
                words = keyword.lower().split()
                for word in words:
                    if (word not in self.stop_words and 
                        len(word) > 1 and 
                        word not in keywords):
                        keywords.append(word)
            
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.warning(f"KeyBERT关键词提取失败: {e}，回退到NLTK方法")
            return self._extract_keywords_nltk(text, max_keywords)
    
    def _extract_keywords_llm(self, text: str, max_keywords: int) -> List[str]:
        """使用大语言模型提取关键词"""
        try:
            if self.llm_client is None:
                # 回退到基于规则的方法
                return self._extract_keywords_with_rules(text, max_keywords)
            
            # 构建提示词
            prompt = f"""
从以下文本中提取最重要的{max_keywords}个关键词，主要关注名词和描述性词汇：

文本: "{text}"

请只返回关键词列表，用逗号分隔，不要包含其他解释：
"""
            
            # 调用OpenAI GPT
            response = self.llm_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的关键词提取助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            # 解析响应
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip().lower() for kw in keywords_text.split(',')]
            
            # 过滤和清理
            filtered_keywords = []
            for kw in keywords:
                if (kw not in self.stop_words and 
                    len(kw) > 1 and 
                    kw.isalpha()):
                    filtered_keywords.append(kw)
            
            return filtered_keywords[:max_keywords]
            
        except Exception as e:
            logger.warning(f"LLM关键词提取失败: {e}，回退到规则方法")
            return self._extract_keywords_with_rules(text, max_keywords)
    
    def _extract_keywords_with_rules(self, text: str, max_keywords: int) -> List[str]:
        """基于规则的关键词提取（作为LLM的回退方法）"""
        # 使用NLTK作为回退
        return self._extract_keywords_nltk(text, max_keywords)
    
    # ==================== 深度学习主词提取方法 ====================
    
    def _extract_main_subject_bert(self, text: str) -> str:
        """使用BERT提取主要中心词"""
        try:
            keywords = self._extract_keywords_bert(text, 10)
            
            # 使用NLTK辅助识别名词
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # 优先选择既是关键词又是名词的词
            for keyword in keywords:
                for word, pos in pos_tags:
                    if (word == keyword and 
                        pos.startswith('NN') and 
                        len(word) > 1):
                        return word
            
            # 如果没有找到，返回第一个关键词
            return keywords[0] if keywords else ""
            
        except Exception as e:
            logger.warning(f"BERT主词提取失败: {e}，回退到NLTK方法")
            return self._extract_main_subject_nltk(text)
    
    def _extract_main_subject_transformer(self, text: str) -> str:
        """使用Transformer提取主要中心词"""
        try:
            keywords = self._extract_keywords_transformer(text, 10)
            
            # 结合词性标注选择最佳主词
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # 寻找既是关键词又是名词的词
            for keyword in keywords:
                for word, pos in pos_tags:
                    if (word == keyword and 
                        pos.startswith('NN')):
                        return word
            
            return keywords[0] if keywords else ""
            
        except Exception as e:
            logger.warning(f"Transformer主词提取失败: {e}，回退到NLTK方法")
            return self._extract_main_subject_nltk(text)
    
    def _extract_main_subject_keybert(self, text: str) -> str:
        """使用KeyBERT提取主要中心词"""
        try:
            keywords = self._extract_keywords_keybert(text, 10)
            
            # 使用词性标注辅助
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # 优先选择名词
            for keyword in keywords:
                for word, pos in pos_tags:
                    if (word == keyword and 
                        pos.startswith('NN')):
                        return word
            
            return keywords[0] if keywords else ""
            
        except Exception as e:
            logger.warning(f"KeyBERT主词提取失败: {e}，回退到NLTK方法")
            return self._extract_main_subject_nltk(text)
    
    def _extract_main_subject_llm(self, text: str) -> str:
        """使用大语言模型提取主要中心词"""
        try:
            if self.llm_client is None:
                return self._extract_main_subject_nltk(text)
            
            prompt = f"""
从以下文本中提取最重要的中心词（主要对象），通常是一个名词：

文本: "{text}"

请只返回一个最重要的中心词，不要包含其他解释：
"""
            
            response = self.llm_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的语言分析助手，专门提取文本的中心词。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            main_subject = response.choices[0].message.content.strip().lower()
            
            # 清理和验证
            if (main_subject and 
                main_subject not in self.stop_words and 
                len(main_subject) > 1 and 
                main_subject.isalpha()):
                return main_subject
            else:
                return self._extract_main_subject_nltk(text)
                
        except Exception as e:
            logger.warning(f"LLM主词提取失败: {e}，回退到NLTK方法")
            return self._extract_main_subject_nltk(text)
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        分析文本，返回完整的分析结果
        
        Args:
            text (str): 输入文本
            
        Returns:
            Dict[str, Any]: 包含关键词、中心词等信息的字典
        """
        result = {
            "original_text": text,
            "method": self.method,
            "keywords": self.extract_keywords(text),
            "main_subject": self.extract_main_subject(text)
        }
        
        return result


def main():
    """测试函数"""
    print("=== 关键词提取器测试 ===")
    
    # 测试用例
    test_cases = [
        "A woman holding an umbrella",
        "A yellow hat carried by a man", 
        "A black cat lying on a chair",
        "Children playing with a ball in the park",
        "A red car parked in the garage",
        "The beautiful sunset over the mountains",
        "A group of students studying in the library"
    ]
    
    # 测试不同的方法
    methods = ['nltk', 'transformer', 'keybert']  # 包括深度学习方法
    
    for method in methods:
        print(f"\n=== 使用 {method.upper()} 方法 ===")
        try:
            extractor = KeywordExtractor(method)
            
            for text in test_cases:
                print(f"\n输入: {text}")
                result = extractor.analyze_text(text)
                print(f"关键词: {result['keywords']}")
                print(f"中心词: {result['main_subject']}")
                
        except Exception as e:
            print(f"方法 {method} 测试失败: {e}")


if __name__ == "__main__":
    main()
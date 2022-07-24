function [TPR, FPR, Pre, Recall, hitRate , falseAlarm] = Pre_Recall_hitRate(testMap,gtMap)
%testMap������ͼ������ֵ�ȽϺ���߼�����
%gtMap����ֵ�߼�����
%�����
neg_gtMap = ~gtMap; %ȡ�෴��
neg_testMap = ~testMap;

hitCount = sum(sum(testMap.*gtMap));%��ֵ�ָ���ͼ����ռ��ֵ�е�Ԫ�أ�1���ĸ���������
trueAvoidCount = sum(sum(neg_testMap.*neg_gtMap));%�Ȳ�������ֵҲ�����ڶ�ֵ���ͼ���Ԫ�صĸ�����1��
missCount = sum(sum(testMap.*neg_gtMap));%��ֵ�ָ��ͼ���д��ֵ�1�ĸ���
falseAvoidCount = sum(sum(neg_testMap.*gtMap));%��ֵ����ֵ��û��û��⵽�ĸ���

if hitCount==0
    Pre = 0;
    Recall = 0;
else
    Pre = hitCount/(hitCount + missCount );
    Recall = hitCount/(hitCount + falseAvoidCount);
end

TPR = Recall;
FPR = missCount/(trueAvoidCount + missCount);
falseAlarm = 1 - trueAvoidCount / (eps+trueAvoidCount + missCount);
hitRate = hitCount / (eps+ hitCount + falseAvoidCount);
end






'use client';

import React, { useState, useRef, useCallback } from "react"
import { Header } from "@/components/Header"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { InsightRadarChart } from "@/components/InsightRadarChart"
import {
    Upload,
    FileText,
    Loader2,
    FolderSearch,
    RefreshCw,
    AlertCircle,
    CheckCircle2,
    Sparkles,
    BookOpen,
    Brain,
    BarChart3,
    FileCheck,
} from "lucide-react"

import {
    analyzeLocalPaper,
    type Step1Data,
    type Step2Data,
    type Step3Data,
    type Step4Data,
} from "@/lib/local-api"

// ============== 步骤状态 ==============
type StepStatus = 'pending' | 'running' | 'done' | 'error';

interface StepState {
    status: StepStatus;
    message: string;
}

export function LocalAnalysisPage() {
    // --- 输入状态 ---
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [referencesDir, setReferencesDir] = useState("")
    const fileInputRef = useRef<HTMLInputElement>(null)

    // --- 分析状态 ---
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [activeTab, setActiveTab] = useState("step1")
    const [error, setError] = useState<string | null>(null)
    const abortRef = useRef<AbortController | null>(null)

    // --- 四个步骤的状态 ---
    const [steps, setSteps] = useState<Record<string, StepState>>({
        step1: { status: 'pending', message: '' },
        step2: { status: 'pending', message: '' },
        step3: { status: 'pending', message: '' },
        step4: { status: 'pending', message: '' },
    })

    // --- 四个步骤的数据 ---
    const [step1Data, setStep1Data] = useState<Step1Data | null>(null)
    const [step2Data, setStep2Data] = useState<Step2Data | null>(null)
    const [step3Data, setStep3Data] = useState<Step3Data | null>(null)
    const [step4Data, setStep4Data] = useState<Step4Data | null>(null)

    // 更新单个步骤的状态
    const updateStep = useCallback((step: string, status: StepStatus, message: string) => {
        setSteps(prev => ({ ...prev, [step]: { status, message } }))
    }, [])

    // 重置
    const handleReset = () => {
        abortRef.current?.abort()
        abortRef.current = null
        setSelectedFile(null)
        setIsAnalyzing(false)
        setActiveTab("step1")
        setError(null)
        setSteps({
            step1: { status: 'pending', message: '' },
            step2: { status: 'pending', message: '' },
            step3: { status: 'pending', message: '' },
            step4: { status: 'pending', message: '' },
        })
        setStep1Data(null)
        setStep2Data(null)
        setStep3Data(null)
        setStep4Data(null)
        if (fileInputRef.current) fileInputRef.current.value = ""
    }

    // 选择文件
    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) {
            setSelectedFile(file)
            setError(null)
        }
    }

    // 开始分析
    const handleStartAnalysis = () => {
        if (!selectedFile) {
            setError("请先选择待评估论文 PDF 文件")
            return
        }
        if (!referencesDir.trim()) {
            setError("请输入引用论文文件夹路径")
            return
        }

        setError(null)
        setIsAnalyzing(true)
        setActiveTab("step1")
        setSteps({
            step1: { status: 'running', message: '正在解析 PDF...' },
            step2: { status: 'pending', message: '' },
            step3: { status: 'pending', message: '' },
            step4: { status: 'pending', message: '' },
        })
        setStep1Data(null)
        setStep2Data(null)
        setStep3Data(null)
        setStep4Data(null)

        const controller = analyzeLocalPaper(selectedFile, referencesDir.trim(), {
            onProgress: (data) => {
                const stepKey = `step${data.step}`
                updateStep(stepKey, 'running', data.message)
            },
            onStep1: (data) => {
                setStep1Data(data)
                updateStep('step1', 'done', `提取到 ${data.total_sentences} 个句子，其中 ${data.cited_sentences} 个带引用`)
                setActiveTab("step1")
                // 开始 step2
                updateStep('step2', 'running', '正在筛选观点句...')
            },
            onStep2: (data) => {
                setStep2Data(data)
                updateStep('step2', 'done', `筛选出 ${data.total_viewpoints} 个观点句`)
                setActiveTab("step2")
                updateStep('step3', 'running', '正在评分...')
            },
            onStep3: (data) => {
                setStep3Data(data)
                updateStep('step3', 'done', `平均分: ${data.avg_score.toFixed(2)}`)
                setActiveTab("step3")
                updateStep('step4', 'running', '正在生成报告...')
            },
            onStep4: (data) => {
                setStep4Data(data)
                updateStep('step4', 'done', `总评分: ${data.overall_score.toFixed(1)}`)
                setActiveTab("step4")
            },
            onDone: () => {
                setIsAnalyzing(false)
            },
            onError: (data) => {
                setError(data.message)
                setIsAnalyzing(false)
            },
        })

        abortRef.current = controller
    }

    // 获取步骤图标
    const getStepIcon = (stepKey: string) => {
        const status = steps[stepKey]?.status
        if (status === 'running') return <Loader2 className="h-4 w-4 animate-spin" />
        if (status === 'done') return <CheckCircle2 className="h-4 w-4 text-green-500" />
        if (status === 'error') return <AlertCircle className="h-4 w-4 text-red-500" />

        switch (stepKey) {
            case 'step1': return <FileText className="h-4 w-4" />
            case 'step2': return <Brain className="h-4 w-4" />
            case 'step3': return <BarChart3 className="h-4 w-4" />
            case 'step4': return <FileCheck className="h-4 w-4" />
            default: return null
        }
    }

    // 计算总进度
    const completedSteps = Object.values(steps).filter(s => s.status === 'done').length
    const progressPercent = isAnalyzing
        ? Math.max(5, completedSteps * 25)
        : completedSteps === 4 ? 100 : 0

    // 获取洞察力等级颜色
    const getLevelBadge = (level: string) => {
        switch (level) {
            case 'high': return <Badge className="text-xs bg-green-500/15 text-green-600 border-green-200">High</Badge>
            case 'medium': return <Badge className="text-xs bg-amber-500/15 text-amber-600 border-amber-200">Medium</Badge>
            case 'low': return <Badge className="text-xs bg-red-500/15 text-red-600 border-red-200">Low</Badge>
            default: return <Badge variant="secondary" className="text-xs">{level}</Badge>
        }
    }

    return (
        <div className="min-h-screen bg-background">
            <Header />
            <main className="py-8 md:py-12">
                <div className="container px-4 md:px-6">

                    {/* Page Header */}
                    <div className="mx-auto max-w-4xl text-center mb-10">
                        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4">
                            <BookOpen className="h-4 w-4" />
                            InsightEval
                        </div>
                        <h1 className="text-3xl font-bold tracking-tight text-foreground md:text-4xl mb-4 text-balance">
                            Automated Insightfulness Evaluation for Scientific Papers
                        </h1>
                        <p className="text-lg text-muted-foreground text-pretty max-w-2xl mx-auto">
                            Upload your paper and specify a local folder containing reference PDFs (e.g. [1].pdf, [2].pdf).
                            The system will extract evidence from references and evaluate insight depth in real-time.
                        </p>
                    </div>

                    {/* Upload & Config Area */}
                    <div className="mx-auto max-w-5xl mb-8">
                        <Card className="border-2 border-dashed border-border/60 bg-muted/20 p-8">
                            <div className="space-y-6">
                                {/* PDF Upload */}
                                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                                    <div className="flex items-center gap-2 shrink-0 w-40">
                                        <Upload className="h-5 w-5 text-primary" />
                                        <span className="text-sm font-medium text-foreground">Paper PDF</span>
                                    </div>
                                    <div className="flex-1 flex items-center gap-3">
                                        <input
                                            ref={fileInputRef}
                                            type="file"
                                            accept=".pdf"
                                            onChange={handleFileSelect}
                                            className="hidden"
                                            id="local-file-upload"
                                        />
                                        <Button
                                            variant="outline"
                                            className="gap-2 bg-transparent"
                                            onClick={() => fileInputRef.current?.click()}
                                            disabled={isAnalyzing}
                                        >
                                            <FileText className="h-4 w-4" />
                                            {selectedFile ? selectedFile.name : "Browse Files"}
                                        </Button>
                                        {selectedFile && (
                                            <span className="text-xs text-muted-foreground">
                                                ({(selectedFile.size / 1024 / 1024).toFixed(1)} MB)
                                            </span>
                                        )}
                                    </div>
                                </div>

                                {/* References Dir */}
                                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                                    <div className="flex items-center gap-2 shrink-0 w-40">
                                        <FolderSearch className="h-5 w-5 text-primary" />
                                        <span className="text-sm font-medium text-foreground">References Dir</span>
                                    </div>
                                    <div className="flex-1">
                                        <input
                                            type="text"
                                            value={referencesDir}
                                            onChange={(e) => setReferencesDir(e.target.value)}
                                            placeholder="e.g. D:/papers/references (folder containing [1].pdf, [2].pdf ...)"
                                            className="w-full px-4 py-2 text-sm rounded-lg border border-border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
                                            disabled={isAnalyzing}
                                        />
                                    </div>
                                </div>

                                {/* Error */}
                                {error && (
                                    <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 px-4 py-2 rounded-lg">
                                        <AlertCircle className="h-4 w-4 shrink-0" />
                                        {error}
                                    </div>
                                )}

                                {/* Action buttons */}
                                <div className="flex items-center gap-3">
                                    <Button
                                        className="gap-2"
                                        onClick={handleStartAnalysis}
                                        disabled={isAnalyzing || !selectedFile}
                                    >
                                        {isAnalyzing ? (
                                            <>
                                                <Loader2 className="h-4 w-4 animate-spin" />
                                                Analyzing...
                                            </>
                                        ) : (
                                            <>
                                                <Sparkles className="h-4 w-4" />
                                                Start Analysis
                                            </>
                                        )}
                                    </Button>
                                    {(step1Data || isAnalyzing) && (
                                        <Button variant="outline" className="gap-2 bg-transparent" onClick={handleReset}>
                                            <RefreshCw className="h-4 w-4" />
                                            Reset
                                        </Button>
                                    )}
                                </div>

                                {/* Progress */}
                                {isAnalyzing && (
                                    <div className="space-y-2">
                                        <Progress value={progressPercent} className="h-2" />
                                        <div className="flex items-center gap-6 text-xs text-muted-foreground">
                                            {(['step1', 'step2', 'step3', 'step4'] as const).map((key, i) => (
                                                <div key={key} className="flex items-center gap-1.5">
                                                    {getStepIcon(key)}
                                                    <span className={steps[key].status === 'running' ? 'text-primary font-medium' : ''}>
                                                        Step {i + 1}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </Card>
                    </div>

                    {/* Results Area — 4 Tabs */}
                    {(step1Data || isAnalyzing) && (
                        <div className="mx-auto max-w-7xl">
                            <Tabs value={activeTab} onValueChange={setActiveTab}>
                                <TabsList className="grid w-full grid-cols-4 mb-6">
                                    <TabsTrigger value="step1" className="gap-2">
                                        {getStepIcon('step1')}
                                        <span className="hidden sm:inline">Sentences</span>
                                        <span className="sm:hidden">S1</span>
                                    </TabsTrigger>
                                    <TabsTrigger value="step2" className="gap-2" disabled={!step2Data && steps.step2.status === 'pending'}>
                                        {getStepIcon('step2')}
                                        <span className="hidden sm:inline">Viewpoints</span>
                                        <span className="sm:hidden">S2</span>
                                    </TabsTrigger>
                                    <TabsTrigger value="step3" className="gap-2" disabled={!step3Data && steps.step3.status === 'pending'}>
                                        {getStepIcon('step3')}
                                        <span className="hidden sm:inline">Scoring</span>
                                        <span className="sm:hidden">S3</span>
                                    </TabsTrigger>
                                    <TabsTrigger value="step4" className="gap-2" disabled={!step4Data && steps.step4.status === 'pending'}>
                                        {getStepIcon('step4')}
                                        <span className="hidden sm:inline">Report</span>
                                        <span className="sm:hidden">S4</span>
                                    </TabsTrigger>
                                </TabsList>

                                {/* Step 1: Extracted Sentences */}
                                <TabsContent value="step1">
                                    {steps.step1.status === 'running' && !step1Data ? (
                                        <Card className="p-8">
                                            <div className="flex flex-col items-center text-center">
                                                <Loader2 className="h-10 w-10 animate-spin text-primary mb-4" />
                                                <p className="text-sm text-muted-foreground">{steps.step1.message || '正在解析 PDF...'}</p>
                                            </div>
                                        </Card>
                                    ) : step1Data ? (
                                        <div className="space-y-4">
                                            <Card className="p-4">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                                                        <span className="font-medium text-foreground">Step 1: Sentence Extraction</span>
                                                    </div>
                                                    <div className="flex items-center gap-2">
                                                        <Badge variant="secondary">{step1Data.total_sentences} sentences</Badge>
                                                        <Badge className="bg-primary/10 text-primary border-primary/20">
                                                            {step1Data.cited_sentences} with citations
                                                        </Badge>
                                                    </div>
                                                </div>
                                            </Card>
                                            <Card className="p-0 overflow-hidden">
                                                <div className="max-h-[500px] overflow-y-auto divide-y">
                                                    {step1Data.sentences.map((s, i) => (
                                                        <div
                                                            key={i}
                                                            className={`px-5 py-3 ${s.has_citation
                                                                ? 'bg-blue-500/5 border-l-4 border-l-blue-400'
                                                                : 'border-l-4 border-l-transparent'
                                                                }`}
                                                        >
                                                            <div className="flex items-start gap-3">
                                                                <span className="text-xs font-mono text-muted-foreground w-6 pt-0.5 shrink-0">{i + 1}</span>
                                                                <div className="flex-1">
                                                                    <p className="text-sm leading-relaxed text-foreground">{s.text}</p>
                                                                    <div className="mt-1.5 flex items-center gap-2">
                                                                        {s.has_citation ? (
                                                                            <Badge variant="outline" className="text-xs text-blue-600 border-blue-200">
                                                                                Citation
                                                                            </Badge>
                                                                        ) : (
                                                                            <Badge variant="outline" className="text-xs">Context</Badge>
                                                                        )}
                                                                        {s.citation_numbers.length > 0 && (
                                                                            <span className="text-xs text-muted-foreground">
                                                                                Refs: [{s.citation_numbers.join(', ')}]
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </Card>
                                        </div>
                                    ) : null}
                                </TabsContent>

                                {/* Step 2: Viewpoints + Evidence */}
                                <TabsContent value="step2">
                                    {steps.step2.status === 'running' && !step2Data ? (
                                        <Card className="p-8">
                                            <div className="flex flex-col items-center text-center">
                                                <Loader2 className="h-10 w-10 animate-spin text-primary mb-4" />
                                                <p className="text-sm text-muted-foreground">{steps.step2.message || '正在筛选观点句并提取证据...'}</p>
                                            </div>
                                        </Card>
                                    ) : step2Data ? (
                                        <div className="space-y-4">
                                            <Card className="p-4">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                                                        <span className="font-medium text-foreground">Step 2: Viewpoints & Evidence</span>
                                                    </div>
                                                    <Badge className="bg-primary/10 text-primary border-primary/20">
                                                        {step2Data.total_viewpoints} viewpoints
                                                    </Badge>
                                                </div>
                                            </Card>
                                            <div className="space-y-4">
                                                {step2Data.viewpoints.map((vp) => (
                                                    <Card key={vp.id} className="p-5 space-y-3">
                                                        <div className="flex items-start gap-3">
                                                            <div className="flex items-center justify-center h-7 w-7 rounded-full bg-primary/10 text-xs font-bold text-primary shrink-0 mt-0.5">
                                                                {vp.id}
                                                            </div>
                                                            <div className="flex-1">
                                                                <p className="text-sm font-medium text-foreground leading-relaxed">{vp.text}</p>
                                                                {vp.citation_numbers.length > 0 && (
                                                                    <span className="text-xs text-muted-foreground mt-1 inline-block">
                                                                        Refs: [{vp.citation_numbers.join(', ')}]
                                                                    </span>
                                                                )}
                                                            </div>
                                                        </div>
                                                        {vp.analysis && (
                                                            <p className="text-xs text-muted-foreground italic pl-10">{vp.analysis}</p>
                                                        )}
                                                        {vp.evidence.length > 0 ? (
                                                            <div className="pl-10 space-y-2">
                                                                <span className="text-xs font-semibold text-muted-foreground uppercase">Evidence from References</span>
                                                                {vp.evidence.map((ev, ei) => (
                                                                    <div key={ei} className="rounded-lg border border-dashed p-3 bg-muted/20">
                                                                        <div className="flex items-center gap-2 mb-1">
                                                                            <Badge variant="outline" className="text-xs">{ev.source || 'Unknown'}</Badge>
                                                                            {ev.relevance && (
                                                                                <span className="text-xs text-muted-foreground capitalize">{ev.relevance}</span>
                                                                            )}
                                                                        </div>
                                                                        <p className="text-sm text-foreground">{ev.quote}</p>
                                                                        {ev.explanation && (
                                                                            <p className="text-xs text-muted-foreground mt-1">{ev.explanation}</p>
                                                                        )}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        ) : (
                                                            <p className="text-xs text-muted-foreground pl-10">
                                                                No direct evidence extracted from local references.
                                                            </p>
                                                        )}
                                                    </Card>
                                                ))}
                                                {step2Data.viewpoints.length === 0 && (
                                                    <Card className="p-8 text-center text-muted-foreground">
                                                        <Brain className="h-10 w-10 mx-auto mb-3 opacity-20" />
                                                        <p>No viewpoint sentences identified</p>
                                                    </Card>
                                                )}
                                            </div>
                                        </div>
                                    ) : null}
                                </TabsContent>

                                {/* Step 3: Scoring */}
                                <TabsContent value="step3">
                                    {steps.step3.status === 'running' && !step3Data ? (
                                        <Card className="p-8">
                                            <div className="flex flex-col items-center text-center">
                                                <Loader2 className="h-10 w-10 animate-spin text-primary mb-4" />
                                                <p className="text-sm text-muted-foreground">{steps.step3.message || '正在对观点句进行评分...'}</p>
                                            </div>
                                        </Card>
                                    ) : step3Data ? (
                                        <div className="space-y-4">
                                            <Card className="p-4">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                                                        <span className="font-medium text-foreground">Step 3: Viewpoint Scoring</span>
                                                    </div>
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-sm text-muted-foreground">Avg Score:</span>
                                                        <span className="text-lg font-bold text-primary">{step3Data.avg_score.toFixed(2)}</span>
                                                        <span className="text-sm text-muted-foreground">/5.0</span>
                                                    </div>
                                                </div>
                                            </Card>

                                            {/* Radar Chart for averages */}
                                            {step3Data.scored_viewpoints.length > 0 && (() => {
                                                const avgSyn = step3Data.scored_viewpoints.reduce((a, v) => a + v.scores.synthesis, 0) / step3Data.scored_viewpoints.length
                                                const avgCri = step3Data.scored_viewpoints.reduce((a, v) => a + v.scores.critical, 0) / step3Data.scored_viewpoints.length
                                                const avgAbs = step3Data.scored_viewpoints.reduce((a, v) => a + v.scores.abstraction, 0) / step3Data.scored_viewpoints.length
                                                return (
                                                    <Card className="p-6">
                                                        <h4 className="text-sm font-semibold text-foreground mb-4 text-center">Average Dimension Scores</h4>
                                                        <div className="flex justify-center">
                                                            <InsightRadarChart scores={{
                                                                synthesis: Math.round(avgSyn * 10) / 10,
                                                                critical: Math.round(avgCri * 10) / 10,
                                                                abstraction: Math.round(avgAbs * 10) / 10,
                                                            }} />
                                                        </div>
                                                    </Card>
                                                )
                                            })()}

                                            {/* Individual scored viewpoints */}
                                            <div className="space-y-3">
                                                {step3Data.scored_viewpoints.map((sv) => (
                                                    <Card key={sv.id} className="p-5">
                                                        <div className="flex items-start gap-3">
                                                            <div className="flex items-center justify-center h-7 w-7 rounded-full bg-primary/10 text-xs font-bold text-primary shrink-0 mt-0.5">
                                                                {sv.id}
                                                            </div>
                                                            <div className="flex-1 space-y-2">
                                                                <p className="text-sm text-foreground leading-relaxed">{sv.text}</p>
                                                                <div className="flex items-center gap-3 flex-wrap">
                                                                    <div className="flex items-center gap-1.5">
                                                                        <span className="text-xs text-muted-foreground">Syn:</span>
                                                                        <span className="text-sm font-semibold">{sv.scores.synthesis.toFixed(1)}</span>
                                                                    </div>
                                                                    <div className="flex items-center gap-1.5">
                                                                        <span className="text-xs text-muted-foreground">Crit:</span>
                                                                        <span className="text-sm font-semibold">{sv.scores.critical.toFixed(1)}</span>
                                                                    </div>
                                                                    <div className="flex items-center gap-1.5">
                                                                        <span className="text-xs text-muted-foreground">Abst:</span>
                                                                        <span className="text-sm font-semibold">{sv.scores.abstraction.toFixed(1)}</span>
                                                                    </div>
                                                                    <span className="mx-1 text-border">|</span>
                                                                    {getLevelBadge(sv.insight_level)}
                                                                </div>
                                                                <p className="text-xs text-muted-foreground italic">{sv.analysis}</p>
                                                            </div>
                                                        </div>
                                                    </Card>
                                                ))}
                                            </div>
                                        </div>
                                    ) : null}
                                </TabsContent>

                                {/* Step 4: Report */}
                                <TabsContent value="step4">
                                    {steps.step4.status === 'running' && !step4Data ? (
                                        <Card className="p-8">
                                            <div className="flex flex-col items-center text-center">
                                                <Loader2 className="h-10 w-10 animate-spin text-primary mb-4" />
                                                <p className="text-sm text-muted-foreground">{steps.step4.message || '正在生成洞察力报告...'}</p>
                                            </div>
                                        </Card>
                                    ) : step4Data ? (
                                        <div className="space-y-4">
                                            <Card className="p-4">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                                                        <span className="font-medium text-foreground">Step 4: Insight Report</span>
                                                    </div>
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-2xl font-bold text-primary">{step4Data.overall_score.toFixed(1)}</span>
                                                        <span className="text-muted-foreground">/10</span>
                                                    </div>
                                                </div>
                                            </Card>

                                            <Card className="p-6 space-y-6">
                                                <div>
                                                    <h4 className="text-sm font-semibold text-foreground mb-2">Summary</h4>
                                                    <p className="text-sm text-muted-foreground leading-relaxed">{step4Data.summary}</p>
                                                </div>

                                                <div className="grid md:grid-cols-2 gap-6 pt-4 border-t">
                                                    <div>
                                                        <h4 className="text-sm font-semibold text-green-600 mb-3 flex items-center gap-2">
                                                            <CheckCircle2 className="h-4 w-4" />
                                                            Strengths
                                                        </h4>
                                                        <ul className="space-y-2">
                                                            {step4Data.strengths.map((s, i) => (
                                                                <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                                                                    <span className="h-1.5 w-1.5 rounded-full bg-green-500 mt-1.5 shrink-0" />
                                                                    {s}
                                                                </li>
                                                            ))}
                                                            {step4Data.strengths.length === 0 && (
                                                                <li className="text-sm text-muted-foreground italic">No strengths identified</li>
                                                            )}
                                                        </ul>
                                                    </div>
                                                    <div>
                                                        <h4 className="text-sm font-semibold text-amber-600 mb-3 flex items-center gap-2">
                                                            <AlertCircle className="h-4 w-4" />
                                                            Areas for Improvement
                                                        </h4>
                                                        <ul className="space-y-2">
                                                            {step4Data.weaknesses.map((w, i) => (
                                                                <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                                                                    <span className="h-1.5 w-1.5 rounded-full bg-amber-500 mt-1.5 shrink-0" />
                                                                    {w}
                                                                </li>
                                                            ))}
                                                            {step4Data.weaknesses.length === 0 && (
                                                                <li className="text-sm text-muted-foreground italic">No weaknesses identified</li>
                                                            )}
                                                        </ul>
                                                    </div>
                                                </div>
                                            </Card>
                                        </div>
                                    ) : null}
                                </TabsContent>
                            </Tabs>
                        </div>
                    )}

                </div>
            </main>
        </div>
    )
}

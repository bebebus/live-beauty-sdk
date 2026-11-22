#import "FLTFaceGeometry.h"
#import <math.h>

static double FLTGeometryStep(double low, double high, double value) {
    double t=fmin(fmax((value-low)/(high-low),0),1);
    return t*t*(3-2*t);
}

@implementation FLTFaceGeometry
+ (instancetype)fromMesh:(NSArray<NSArray<NSNumber *> *> *)points aspect:(CGFloat)aspect {
    if (points.count < 468 || !isfinite(aspect) || aspect <= 0) return nil;
    // 既有 9 个塑形点要求在 [0, 1]；新增 8 个遮罩点允许 [-0.25, 1.25]。任一点非有限或越界即全部作废。
    NSArray *indexes=@[@132,@361,@172,@397,@33,@133,@362,@263,@1];
    NSArray *maskIndexes=@[@234,@454,@10,@152,@61,@291,@0,@17];
    for (NSArray *group in @[indexes,maskIndexes]) {
        double low=group == indexes ? 0 : -0.25, high=group == indexes ? 1 : 1.25;
        for (NSNumber *index in group) {
            NSArray *point=points[index.integerValue];
            if (point.count < 2) return nil;
            for (NSUInteger i=0;i<2;i++) {
                double v=[point[i] doubleValue];
                if (!isfinite(v) || v < low || v > high) return nil;
            }
        }
    }
    CGPoint (^p)(NSInteger)=^CGPoint(NSInteger i) {
        return CGPointMake(points[i][0].doubleValue,1-points[i][1].doubleValue);
    };
    CGFloat (^distance)(CGPoint,CGPoint)=^CGFloat(CGPoint a,CGPoint b) {
        return hypot(a.x-b.x,(a.y-b.y)/aspect);
    };
    CGPoint left=p(132), right=p(361), nose=p(1);
    CGFloat span=distance(left,right);
    if (span < 0.1 || span > 0.85) return nil;
    CGPoint center=CGPointMake((left.x+right.x)/2,(left.y+right.y)/2);
    CGFloat yaw=fabs((nose.x-center.x)*(right.x-left.x)+
        (nose.y-center.y)*(right.y-left.y)/(aspect*aspect))/span;
    // 侧脸比 r ≤ 0.45（约 28°）时几何有效；塑形在 0.24..0.30 之间渐出。
    CGFloat ratio=yaw/span;
    if (ratio > 0.45) return nil;
    NSMutableArray *cheeks=[NSMutableArray new], *eyes=[NSMutableArray new];
    for (NSArray *pair in @[@[@132,@361],@[@172,@397]]) {
        CGPoint a=p([pair[0] integerValue]), b=p([pair[1] integerValue]);
        CGFloat x=(a.x+b.x)/2,y=(a.y+b.y)/2;
        [cheeks addObject:[CIVector vectorWithX:a.x Y:a.y Z:x W:y]];
        [cheeks addObject:[CIVector vectorWithX:b.x Y:b.y Z:x W:y]];
    }
    for (NSArray *pair in @[@[@33,@133],@[@362,@263]]) {
        CGPoint a=p([pair[0] integerValue]), b=p([pair[1] integerValue]);
        CGFloat width=distance(a,b);
        if (width < 0.015 || width > span*0.5) return nil;
        [eyes addObject:[CIVector vectorWithX:(a.x+b.x)/2 Y:(a.y+b.y)/2 Z:width*0.9]];
    }
    // 以下在度量空间 p' = (x, y / aspect) 中计算。
    CGPoint (^m)(NSInteger)=^CGPoint(NSInteger i) {
        CGPoint t=p(i);
        return CGPointMake(t.x,t.y/aspect);
    };
    CGFloat (^dot)(CGPoint,CGPoint)=^CGFloat(CGPoint a,CGPoint b) { return a.x*b.x+a.y*b.y; };
    CGPoint (^sub)(CGPoint,CGPoint)=^CGPoint(CGPoint a,CGPoint b) { return CGPointMake(a.x-b.x,a.y-b.y); };
    CGPoint (^mid)(CGPoint,CGPoint)=^CGPoint(CGPoint a,CGPoint b) { return CGPointMake((a.x+b.x)/2,(a.y+b.y)/2); };
    CGPoint p234=m(234), p454=m(454), p10=m(10), p152=m(152);
    CGPoint widthVector=sub(p454,p234);
    CGFloat faceWidth=hypot(widthVector.x,widthVector.y);
    if (faceWidth < 0.05) return nil;
    CGPoint u=CGPointMake(widthVector.x/faceWidth,widthVector.y/faceWidth);
    CGPoint v=CGPointMake(-u.y,u.x);
    if (dot(v,sub(p10,p152)) < 0) { // v 必须指向额头（152 → 10）
        u=CGPointMake(-u.x,-u.y);
        v=CGPointMake(-u.y,u.x);
    }
    if (!isfinite(u.x) || !isfinite(u.y) || fabs(hypot(u.x,u.y)-1) >= 1e-6) return nil;
    // u 分量取侧向点对中点与中线点对中点的平均，v 分量只取中线点对中点；返回纹理坐标。
    CGPoint (^centre)(CGPoint,CGPoint,CGPoint,CGPoint)=^CGPoint(CGPoint sideA,CGPoint sideB,CGPoint midA,CGPoint midB) {
        CGPoint sm=mid(sideA,sideB), mm=mid(midA,midB);
        CGFloat cu=0.5*(dot(sm,u)+dot(mm,u)), cv=dot(mm,v);
        return CGPointMake(cu*u.x+cv*v.x,(cu*u.y+cv*v.y)*aspect);
    };
    CGPoint ovalCentre=centre(p234,p454,p10,p152);
    CGFloat ovalRx=0.5*faceWidth*1.12, ovalRy=0.5*fabs(dot(sub(p10,p152),v))*1.15;
    if (ovalRx < 0.03 || ovalRy < 0.03) return nil;
    CGPoint p61=m(61), p291=m(291), p0=m(0), p17=m(17);
    CGPoint mouthCentre=centre(p61,p291,p0,p17);
    CGFloat mouthRx=0.5*fabs(dot(sub(p61,p291),u))*1.20;
    if (mouthRx < 0.005) return nil;
    CGFloat mouthRy=fmax(0.5*fabs(dot(sub(p0,p17),v))*1.25,0.25*mouthRx);
    NSMutableArray *eyeMasks=[NSMutableArray new];
    for (NSArray *pair in @[@[@33,@133],@[@362,@263]]) {
        CGPoint a=m([pair[0] integerValue]), b=m([pair[1] integerValue]);
        CGFloat width=hypot(a.x-b.x,a.y-b.y);
        CGPoint c=mid(a,b);
        CGFloat x=c.x+0.04*width*v.x, y=c.y+0.04*width*v.y;
        [eyeMasks addObject:[CIVector vectorWithX:x Y:y*aspect Z:0.72*width W:0.36*width]];
    }
    FLTFaceGeometry *geometry=[self new]; geometry.cheeks=cheeks; geometry.eyes=eyes;
    geometry.oval=[CIVector vectorWithX:ovalCentre.x Y:ovalCentre.y Z:ovalRx W:ovalRy];
    geometry.axis=[CIVector vectorWithX:u.x Y:u.y];
    geometry.mouth=[CIVector vectorWithX:mouthCentre.x Y:mouthCentre.y Z:mouthRx W:mouthRy];
    geometry.eyeMasks=eyeMasks;
    geometry.warpWeight=1-FLTGeometryStep(0.24,0.30,ratio);
    return geometry;
}
- (void)getState:(float *)state {
    NSMutableArray<CIVector *> *vectors=[NSMutableArray arrayWithArray:self.cheeks];
    [vectors addObjectsFromArray:self.eyes];
    [vectors addObjectsFromArray:@[self.oval,self.axis,self.mouth]];
    [vectors addObjectsFromArray:self.eyeMasks];
    int i=0;
    for (CIVector *vector in vectors) for (size_t k=0;k<vector.count;k++) state[i++]=[vector valueAtIndex:k];
    state[i]=self.warpWeight;
}
@end
